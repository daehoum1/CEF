import ast
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Style remapping (many-to-one consolidation for cleaner zero-shot evaluation)
# ──────────────────────────────────────────────────────────────────────────────
STYLE_MAPPING = {
    "Impressionism": "Impressionism",
    "Realism": "Realism",
    "Romanticism": None,
    "Expressionism": "Expressionism",
    "Post Impressionism": "Post Impressionism",
    "Symbolism": "Symbolism",
    "Baroque": "Baroque",
    "Art Nouveau Modern": "Art Nouveau Modern",
    "Abstract Expressionism": "Abstract Expressionism",
    "Northern Renaissance": "Northern Renaissance",
    "Naive Art Primitivism": None,
    "Cubism": "Cubism",
    "Rococo": "Rococo",
    "Color Field Painting": None,
    "Pop Art": "Pop Art",
    "Early Renaissance": "Italian Renaissance",
    "High Renaissance": "Italian Renaissance",
    "Minimalism": "Minimalism",
    "Mannerism Late Renaissance": "Italian Renaissance",
    "Ukiyo e": None,
    "Fauvism": "Expressionism",
    "Pointillism": "Pointillism",
    "Contemporary Realism": "Realism",
    "New Realism": "Realism",
    "Synthetic Cubism": "Cubism",
    "Analytical Cubism": "Cubism",
    "Action painting": "Abstract Expressionism",
}


def _parse_style(x):
    if isinstance(x, list):
        return x[0]
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
        except Exception:
            pass
        return x
    return str(x)


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """Load the WikiArt CSV and apply style remapping. Drops unmapped styles."""
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"[dataset] Loaded CSV ({enc}), {len(df)} rows")
            break
        except Exception:
            df = None

    if df is None:
        raise RuntimeError(f"Could not read {csv_path}")

    df = df.copy()
    df["style_original"] = df["genre"].apply(_parse_style)
    df["style"] = df["style_original"].map(STYLE_MAPPING)
    df["artist"] = df["artist"].astype(str).str.strip().str.lower()
    df = df[df["style"].notna()].reset_index(drop=True)

    print(f"[dataset] After remapping: {len(df)} rows, {df['style'].nunique()} styles")
    return df


def maybe_filter_styles(
    df: pd.DataFrame,
    min_images: int = 0,
    top_n: int = 0,
    selected: list = None,
) -> pd.DataFrame:
    if min_images > 0:
        counts = df["style"].value_counts()
        df = df[df["style"].isin(counts[counts >= min_images].index)].copy()

    if selected:
        df = df[df["style"].isin(selected)].copy()
    elif top_n > 0:
        top = df["style"].value_counts().head(top_n).index.tolist()
        df = df[df["style"].isin(top)].copy()

    return df.reset_index(drop=True)


class WikiArtDataset(Dataset):
    """Returns (PIL Image, label_index, meta_dict) for CLIP-based inference."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        class_names: list,
        preprocess=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_root, row["filename"])
        img = Image.open(path).convert("RGB")
        if self.preprocess is not None:
            img = self.preprocess(img)
        label = self.class_to_idx[row["style"]]
        meta = {
            "filename": row["filename"],
            "artist": row["artist"],
            "style": row["style"],
        }
        return img, label, meta
