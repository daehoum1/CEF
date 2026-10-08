"""
CLIP encoder with multi-template zero-shot scoring.

Supported backends (auto-detected in order):
  1. openai/clip  → pip install git+https://github.com/openai/CLIP.git
  2. open_clip    → pip install open-clip-torch
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.special import softmax

# ──────────────────────────────────────────────────────────────────────────────
# Natural-language display names for each internal style label.
# CLIP text prompts use these names, not the raw dataset labels.
# ──────────────────────────────────────────────────────────────────────────────
STYLE_DISPLAY_NAMES: dict[str, str] = {
    "Abstract Expressionism": "Abstract Expressionism",
    "Art Nouveau Modern": "Art Nouveau",
    "Baroque": "Baroque",
    "Cubism": "Cubism",
    "Expressionism": "Expressionism",
    "Impressionism": "Impressionism",
    "Italian Renaissance": "Italian Renaissance",
    "Minimalism": "Minimalism",
    "Northern Renaissance": "Northern Renaissance",
    "Pointillism": "Pointillism",
    "Pop Art": "Pop Art",
    "Post Impressionism": "Post-Impressionism",
    "Realism": "Realism",
    "Rococo": "Rococo",
    "Symbolism": "Symbolism",
}

# Ensemble of prompt templates (averaged to improve zero-shot accuracy)
PROMPT_TEMPLATES = [
    "a painting in the style of {}",
    "a {} painting",
    "an artwork in the style of {}",
    "{} style painting",
    "a famous {} artwork",
]


def _load_clip_backend(model_name: str, device: str):
    """Return (model, preprocess, tokenize_fn, backend_name)."""
    try:
        import clip  # openai/clip

        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        return model, preprocess, clip.tokenize, "openai-clip"
    except ImportError:
        pass

    try:
        import open_clip

        ocl_name = model_name.replace("/", "-")  # "ViT-B/32" → "ViT-B-32"
        model, _, preprocess = open_clip.create_model_and_transforms(
            ocl_name, pretrained="openai", device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(ocl_name)
        return model, preprocess, tokenizer, "open-clip"
    except ImportError:
        pass

    raise ImportError(
        "No CLIP backend found. Install one of:\n"
        "  pip install git+https://github.com/openai/CLIP.git\n"
        "  pip install open-clip-torch"
    )


class CLIPEncoder:
    def __init__(self, model_name: str = "ViT-B/32", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.preprocess, self._tokenize, self.backend = _load_clip_backend(
            model_name, device
        )
        print(f"[clip] Backend={self.backend}, model={model_name}, device={device}")

    # ──────────────────────────────────────────────────────────────────────────
    # Encoding
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_images_from_paths(
        self, paths: list[str], batch_size: int = 64
    ) -> np.ndarray:
        """Load images from disk, run through CLIP vision encoder.
        Returns L2-normalized embeddings [N, D]."""
        all_feats: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            imgs = [
                self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths
            ]
            imgs_t = torch.stack(imgs).to(self.device)
            feats = self.model.encode_image(imgs_t).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
            if (start // batch_size) % 10 == 0:
                print(f"  encoded {start + len(batch_paths)}/{len(paths)} images", end="\r")
        print()
        return np.concatenate(all_feats, axis=0)

    @torch.no_grad()
    def encode_images_from_tensors(
        self, loader, batch_size: int = 64
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Run inference on a DataLoader that yields (tensor, label, meta).
        Returns (embeddings [N,D], labels [N], metas list)."""
        all_feats, all_labels, all_metas = [], [], []
        for imgs, labels, metas in loader:
            imgs = imgs.to(self.device)
            feats = self.model.encode_image(imgs).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            all_metas.extend(metas)
        return (
            np.concatenate(all_feats, axis=0),
            np.concatenate(all_labels, axis=0),
            all_metas,
        )

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized text embeddings [len(texts), D]."""
        tokens = self._tokenize(texts).to(self.device)
        feats = self.model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────────
    # Zero-shot scoring
    # ──────────────────────────────────────────────────────────────────────────

    def get_text_embeddings(
        self,
        class_names: list[str],
        templates: list[str] = PROMPT_TEMPLATES,
        display_names: dict[str, str] | None = STYLE_DISPLAY_NAMES,
    ) -> np.ndarray:
        """Ensemble text embeddings over multiple prompt templates.
        Returns averaged + re-normalized embeddings [C, D]."""
        resolved = [
            display_names.get(c, c) if display_names else c for c in class_names
        ]
        stacked = []
        for tmpl in templates:
            prompts = [tmpl.format(name) for name in resolved]
            embs = self.encode_texts(prompts)  # [C, D]
            stacked.append(embs)
        # Average and renormalize
        avg = np.mean(np.stack(stacked, axis=0), axis=0)  # [C, D]
        avg = avg / (np.linalg.norm(avg, axis=1, keepdims=True) + 1e-8)
        return avg

    def zero_shot_scores(
        self,
        image_embs: np.ndarray,
        text_embs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute softmax probabilities and raw logits.

        Uses the model's learned logit_scale (≈100) for proper calibration.
        Returns (probs [N,C], logits [N,C]).
        """
        logit_scale = 1.0
        if hasattr(self.model, "logit_scale"):
            logit_scale = self.model.logit_scale.exp().item()

        logits = (image_embs @ text_embs.T) * logit_scale  # [N, C]
        probs = softmax(logits, axis=1)
        return probs, logits
