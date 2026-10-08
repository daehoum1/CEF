# Concept-Evidence Fusion (CEF)

Code for **"Concept-Evidence Fusion for Zero-Shot Style Classification"**
(Yoorim Kim, Jungyeob Han, Daeho Um), under review at IEEE Access.

CEF is a training-free, transductive method for zero-shot painting style
classification with frozen vision-language models. It turns an LLM-generated
concept dictionary into class evidence through two branches — dense
image-phrase affinities and class-level text prototypes — and fuses that evidence
with the zero-shot scores inside a label-propagation graph. "Training-free" means
that no model parameters are trained or fine-tuned; configurations are selected on
a labeled validation split, as described in Section V-A of the paper.

## Contents

| Path | Contents |
|---|---|
| `code/analysis/` | Table generators, numerical verifiers, experiment and preparation scripts |
| `code/src/`, `code/ugsp/`, `code/scripts/`, `code/configs/` | Concept-hypergraph and encoding dependencies used by those scripts |
| `splits/` | The validation/test partition of every seed, by image filename, for both protocols |
| `style_concepts_wikiart.json`, `style_concepts_mp100k.json` | The reference concept dictionaries |

The per-seed outputs of every reported method, the remaining dictionaries
(regenerations, controls, domain probes) and the generation records are in the
reproducibility archive that accompanies the article, which is too large to carry
here.

## Splits

`splits/<dataset>_<protocol>.csv.gz` has the columns
`seed, partition, filename, style, artist`. The `artist_disjoint` files are the
partition of the main tables, in which every artist belongs wholly to validation
or to test; the `image_level` files are the class-stratified image-level
partition used for the secondary comparison under CGPR's protocol. Seeds are 42,
43, 44, 45 and 46 throughout.

## Backbones

Backbones are `open_clip` checkpoints, named by `(model, pretrained)`:

| Paper name | open_clip model | pretrained tag |
|---|---|---|
| CLIP (OpenAI) | `ViT-B-32` | `openai` |
| MetaCLIP | `ViT-B-32` | `metaclip_fullcc` |
| EVA02-CLIP | `EVA02-B-16` | `merged2b_s8b_b131k` |
| SigLIP | `ViT-B-16-SigLIP` | `webli` |

## Reproducing the reported tables

Extract the reproducibility archive and point the entry script at it:

```sh
pip install -r requirements.txt
python code/analysis/reproduce_submission.py \
    --bundle /path/to/supplementary_material \
    --output /tmp/cef-reproduced
```

Use an empty output directory. The command regenerates every table of the paper
from the saved per-seed outputs, runs the numerical verifiers, and compares each
regenerated table against the reported one. It needs no image dataset, no GPU, no
network and no API key, and finishes in a few minutes. Expected result:

```
Reproduced and matched 15 referenced tables.
```

Analysis and verification were run with Python 3.10.13 on Linux (glibc 2.35),
NumPy 1.26.3, pandas 2.3.3 and SciPy 1.15.3; each result folder of the archive
carries an `environment.json` recording the environment of that run.

## Re-running the experiments from the datasets

Image datasets and cached embeddings are not redistributed. The scripts that need
them read their locations from environment variables, so no author-specific paths
remain in the code:

```sh
export CEF_WIKIART_ROOT=/path/to/wikiart        # images + classes_artist_disjoint_fixed.csv
export CEF_MP100K_ROOT=/path/to/multitask_painting
export CEF_WIKIART_CACHE=/path/to/cached_embeddings/wikiart
export CEF_MP100K_CACHE=/path/to/cached_embeddings/mp100k
```

Encode the dictionaries and images with `code/analysis/prepare_current_cef.py`,
then run selection and evaluation with
`code/analysis/current_cef_experiments.py` (`--phase select`, then
`--phase evaluate`). Selection always precedes the corresponding test evaluation,
and every evaluation record stores the hash of the selection it used.

WikiArt is at <https://github.com/cs-chan/ArtGAN/tree/master> and
MultitaskPainting100k at <http://www.ivl.disco.unimib.it/activities/paintings>.
The non-art probes use Oxford-IIIT Pet, DTD, FGVC-Aircraft and EuroSAT (RGB).

## License

Code is released under the MIT License (`LICENSE`). The concept dictionaries and
the split files are released under CC BY 4.0. The datasets keep the licenses of
their original distributors.
