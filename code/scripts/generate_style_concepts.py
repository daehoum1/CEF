"""
generate_style_concepts.py — Reproducible LLM-based concept dictionary generation
====================================================================================

Generates assets/style_concepts_v3.json (or an MP100k equivalent) by prompting an
LLM once per class with a FIXED template (below). No ground-truth image/label
information is given to the LLM — only the class's display name — so this stays
consistent with the "no ground truth used in hyperedge construction" constraint;
the LLM only supplies candidate text phrases, which are later matched to images
purely via VLM image-text similarity, and filtered for discriminability by
filter_concepts.py before use.

Requires: `pip install anthropic` and the ANTHROPIC_API_KEY environment variable.
If no API key is available, this file still documents the exact prompt used —
see PROMPT_TEMPLATE below — so the generation is reproducible by anyone with API
access, even if it was originally produced via direct model invocation instead of
a scripted API call.

Model used in this work: claude-sonnet-5 (Anthropic), temperature=0 (this script's
default) for maximal determinism / reproducibility.

Usage:
  python scripts/generate_style_concepts.py \\
      --styles "Abstract Expressionism" "Art Nouveau" ... \\
      --out assets/style_concepts_v3.json
"""

from __future__ import annotations

import argparse
import json
import os

MODEL = "claude-sonnet-5"
TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are an art history expert helping build a visual concept dictionary "
    "for zero-shot painting style classification."
)

# The ONLY variable part of the prompt is {display_name}. This exact template is
# reused, unmodified, for every class in every dataset.
PROMPT_TEMPLATE = """Style: "{display_name}"

List 10 to 12 short visual concept phrases (3-6 words each) that describe how a
viewer could visually recognize a painting in this style, WITHOUT naming the
style itself. Cover a mix of:
- brushwork / paint application technique
- color palette / use of light
- composition / form / spatial treatment
- recurring subject matter or motifs
- 1-2 representative artists or landmark works associated with the style
  (named explicitly)

Do not include the style name or a direct synonym of it in any phrase.
Output ONLY a JSON array of strings, nothing else."""


def call_llm(display_name: str) -> list[str]:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(display_name=display_name)}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def run(styles: list[str], out_path: str) -> None:
    result = {
        "_meta": {
            "description": "LLM-generated visual-concept profiles, one fixed prompt per class.",
            "model": MODEL,
            "temperature": TEMPERATURE,
            "system_prompt": SYSTEM_PROMPT,
            "prompt_template": PROMPT_TEMPLATE,
        },
    }
    for style in styles:
        print(f"[generate] {style}")
        result[style] = call_llm(style)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {out_path}")


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--styles", type=str, nargs="+", required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args.styles, args.out)
