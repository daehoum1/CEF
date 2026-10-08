"""Location resolution for this repository.

The released scripts were written against the authors' working tree. This module
replaces the absolute paths they contained, so that every script resolves its
inputs from the repository itself, or from an environment variable when the input
is not redistributable (image datasets and cached embeddings).

    CEF_REPO_ROOT      repository root (default: two levels above this file)
    CEF_PROJECT_ROOT   root of a reproduction output tree, as created by
                       code/analysis/reproduce_submission.py
    CEF_WIKIART_ROOT   WikiArt images and class CSV
    CEF_MP100K_ROOT    MultitaskPainting100k images and CSV
    CEF_WIKIART_CACHE  cached WikiArt embeddings and zero-shot scores
    CEF_MP100K_CACHE   cached MultitaskPainting100k embeddings and scores

Table regeneration and the numerical verifiers need none of the dataset
variables; only re-encoding and re-running the search from raw data do.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("CEF_REPO_ROOT", Path(__file__).resolve().parents[2]))
CODE_ROOT = REPO_ROOT / "code"


def _opt(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return str(Path(value).expanduser()) if value else None


# Bundled sources: the `ugsp` package and `concept_hypergraph` module.
UGSP_ROOT = _opt("CEF_UGSP_ROOT", str(CODE_ROOT))
SRC_ROOT = str(CODE_ROOT / "src")
SCRIPTS_ROOT = str(CODE_ROOT / "scripts")

# Not redistributable: datasets and cached embeddings.
WIKIART_ROOT = _opt("CEF_WIKIART_ROOT")
MP100K_ROOT = _opt("CEF_MP100K_ROOT")
WIKIART_CACHE = _opt("CEF_WIKIART_CACHE")
MP100K_CACHE = _opt("CEF_MP100K_CACHE")

WIKIART_CSV = os.path.join(WIKIART_ROOT, "classes_artist_disjoint_fixed.csv") if WIKIART_ROOT else ""
MP100K_CSV = os.path.join(MP100K_ROOT, "multitask_painting100k.csv") if MP100K_ROOT else ""
MP100K_IMAGES = os.path.join(MP100K_ROOT, "images") if MP100K_ROOT else ""


def add_bundled_paths() -> None:
    """Put the bundled `ugsp` package and `concept_hypergraph` on sys.path."""
    for p in (UGSP_ROOT, SRC_ROOT, SCRIPTS_ROOT):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def project_root() -> Path:
    """Root of the tree a script reads results from (a reproduction output tree)."""
    return Path(os.environ.get("CEF_PROJECT_ROOT", REPO_ROOT))


def require(name: str, value: str | None) -> str:
    if not value:
        raise SystemExit(
            f"{name} is not set. This script needs the image datasets or cached "
            f"embeddings, which are not redistributed with this repository. "
            f"See the Reproduction section of README.md."
        )
    return value
