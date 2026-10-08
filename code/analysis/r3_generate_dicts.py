"""Review round 3: independently generated concept dictionaries.

Five independent generations per style dataset with the exact system prompt and
PROMPT_TEMPLATE recorded in assets/style_concepts_v3.json and
scripts/generate_style_concepts.py. Measures how much CEF depends on one draw of
the dictionary.

Route. No ANTHROPIC_API_KEY exists in this environment, so each call is a fresh
headless Claude Code invocation with Claude Sonnet 5 (the model of the original
dictionary), its system prompt REPLACED by the dictionary system prompt, no tools
and no session persistence, run from an empty directory. Every call is
independent. Sampling temperature cannot be set on this route and is the service
default; the full CLI response, including per-model token usage, is saved per
style for provenance.

Outputs
  assets/r3_dicts/{dataset}_{family}_g{k}.json         dictionary + _meta
  assets/r3_dicts/raw/{dataset}_{family}_g{k}/*.json   every attempt, verbatim
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cef_paths

HYPER = str(cef_paths.project_root())
sys.path.insert(0, os.path.join(HYPER, "analysis"))

OUT = os.path.join(HYPER, "assets", "r3_dicts")
RUNDIR = "/tmp/claude-0/-workspace/1caafd43-dc02-435b-8cda-518cf809a6e2/scratchpad/gen"
MODEL = "claude-sonnet-5"
MAX_TRIES = 6
N_GEN = 5

SYSTEM = {
    "style": "You are an art history expert helping build a visual concept dictionary "
             "for zero-shot painting style classification.",
}
TEMPLATE = {}
TEMPLATE["orig"] = """Style: "{display_name}"

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


V3 = {"wikiart": os.path.join(HYPER, "assets", "style_concepts_v3.json"),
      "mp100k": os.path.join(HYPER, "assets", "style_concepts_mp100k_v3.json")}


def display_names(dataset):
    """The display names the v3 dictionary was generated from (its keys)."""
    return [k for k in json.load(open(V3[dataset])) if k != "_meta"]


def call(system, prompt):
    cmd = ["claude", "-p", "--model", MODEL, "--system-prompt", system, "--tools", "",
           "--no-session-persistence", "--output-format", "json", prompt]
    for attempt in range(5):
        try:
            r = subprocess.run(cmd, cwd=RUNDIR, capture_output=True, text=True, timeout=300)
            d = json.loads(r.stdout)
            if not d.get("is_error") and MODEL in d.get("modelUsage", {}):
                return d
        except Exception as e:  # noqa: BLE001
            d = {"error": repr(e)}
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"CLI failed: {d}")


def parse(text):
    t = text.strip()
    m = re.search(r"\[.*\]", t, re.S)
    arr = json.loads(m.group(0))
    assert isinstance(arr, list) and all(isinstance(x, str) for x in arr)
    return [x.strip() for x in arr if x.strip()]


def one_style(dataset, family, gen, name):
    rawdir = os.path.join(OUT, "raw", f"{dataset}_{family}_g{gen}")
    os.makedirs(rawdir, exist_ok=True)
    fname = os.path.join(rawdir, re.sub(r"[^A-Za-z0-9]+", "_", name) + ".json")
    if os.path.exists(fname):
        rec = json.load(open(fname))
        if "phrases" in rec:
            return name, rec
    prompt = TEMPLATE[family].format(display_name=name)
    attempts = []
    for t in range(MAX_TRIES):
        d = call(SYSTEM["style"], prompt)
        entry = dict(result=d.get("result"), modelUsage=d.get("modelUsage"))
        try:
            phrases = parse(d["result"])
        except Exception as e:  # noqa: BLE001
            entry["rejected"] = f"unparseable: {e!r}"
            attempts.append(entry)
            continue
        if not 10 <= len(phrases) <= 12:
            entry["rejected"] = f"{len(phrases)} phrases"
        attempts.append(entry)
        if "rejected" not in entry:
            break
    else:
        raise RuntimeError(f"{dataset}/{name}: no valid answer after {MAX_TRIES} tries")
    rec = dict(style=name, dataset=dataset, family=family, generation=gen,
               n_attempts=len([a for a in attempts if "result" in a]), attempts=attempts,
               phrases=phrases)
    json.dump(rec, open(fname, "w"), indent=1, ensure_ascii=False)
    return name, rec


def build(dataset, family, gen, workers):
    names = display_names(dataset)
    with ThreadPoolExecutor(workers) as ex:
        recs = dict(ex.map(lambda n: one_style(dataset, family, gen, n), names))
    out = {"_meta": dict(
        dataset=dataset, family=family, generation=gen, model=MODEL,
        system_prompt=SYSTEM["style"], prompt_template=TEMPLATE[family],
        route="headless Claude Code CLI (claude -p), system prompt replaced, no tools, "
              "no session persistence, one independent call per style; service-default "
              "sampling",
        attempts_per_style={n: recs[n]["n_attempts"] for n in names})}
    for n in names:
        out[n] = recs[n]["phrases"]
    path = os.path.join(OUT, f"{dataset}_{family}_g{gen}.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    tot = sum(len(out[n]) for n in names)
    extra = sum(recs[n]["n_attempts"] - 1 for n in names)
    print(f"[saved] {os.path.basename(path)}  {tot} phrases, {extra} re-queries", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--families", nargs="+", default=["orig"])
    a = ap.parse_args()
    os.makedirs(RUNDIR, exist_ok=True)
    for gen in range(N_GEN):
        for dataset in ["wikiart", "mp100k"]:
            for family in a.families:
                build(dataset, family, gen, a.workers)


if __name__ == "__main__":
    main()
