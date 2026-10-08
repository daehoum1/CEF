"""Evaluation utilities for zero-shot classification."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score as sklearn_f1


# ──────────────────────────────────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────────────────────────────────

def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(preds == labels))


def top_k_accuracy(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of samples where the true label is in the top-k predictions."""
    topk = np.argsort(scores, axis=1)[:, -k:]
    hits = [labels[i] in topk[i] for i in range(len(labels))]
    return float(np.mean(hits))


def per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, float | None]:
    result = {}
    for idx, name in enumerate(class_names):
        mask = labels == idx
        result[name] = float(np.mean(preds[mask] == idx)) if mask.sum() > 0 else None
    return result


def mean_per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> float:
    """Macro-averaged per-class accuracy (kept for backward compatibility)."""
    values = [
        v for v in per_class_accuracy(preds, labels, class_names).values()
        if v is not None
    ]
    return float(np.mean(values))


def g_means(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> float:
    """Geometric mean of per-class recalls."""
    recalls = [
        v for v in per_class_accuracy(preds, labels, class_names).values()
        if v is not None
    ]
    if not recalls:
        return 0.0
    return float(np.exp(np.mean(np.log(np.array(recalls) + 1e-10))))


def balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Balanced accuracy = macro-averaged recall."""
    return float(balanced_accuracy_score(labels, preds))


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged F1 score."""
    return float(sklearn_f1(labels, preds, average="macro", zero_division=0))


# ──────────────────────────────────────────────────────────────────────────────
# Multi-method evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    preds = np.argmax(scores, axis=1)
    metrics: dict = {
        "g_means":   g_means(preds, labels, class_names),
        "bacc":      balanced_accuracy(preds, labels),
        "macro_f1":  macro_f1(preds, labels),
    }
    for k in top_ks:
        if k <= scores.shape[1]:
            metrics[f"top{k}"] = top_k_accuracy(scores, labels, k)
    metrics["per_class"] = per_class_accuracy(preds, labels, class_names)
    return metrics


def compare_methods(
    method_scores: dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: list[str],
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, dict]:
    """Evaluate a dict of {method_name: score_matrix} and return all metrics."""
    return {
        name: evaluate_scores(scores, labels, class_names, top_ks)
        for name, scores in method_scores.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single-run reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 86)
    print("ZERO-SHOT STYLE CLASSIFICATION — METHOD COMPARISON")
    print("=" * 86)
    header = (
        f"{'Method':<28} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7}"
        f" {'GMeans':>8} {'bAcc':>7} {'MacroF1':>8}"
    )
    print(header)
    print("-" * 86)
    for method, m in results.items():
        top1   = m.get("top1",     float("nan"))
        top3   = m.get("top3",     float("nan"))
        top5   = m.get("top5",     float("nan"))
        gm     = m.get("g_means",  float("nan"))
        ba     = m.get("bacc",     float("nan"))
        mf1    = m.get("macro_f1", float("nan"))
        print(
            f"{method:<28} {top1:>7.4f} {top3:>7.4f} {top5:>7.4f}"
            f" {gm:>8.4f} {ba:>7.4f} {mf1:>8.4f}"
        )
    print("=" * 86)


def save_results(
    results: dict[str, dict],
    labels: np.ndarray,
    class_names: list[str],
    save_dir: str,
    extra: dict | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    summary = {}
    for method, m in results.items():
        summary[method] = {k: v for k, v in m.items() if k != "per_class"}
    if extra:
        summary["_meta"] = extra
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for method, m in results.items():
        for cls, acc in m.get("per_class", {}).items():
            rows.append({"method": method, "class": cls, "accuracy": acc})
    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, "per_class_accuracy.csv"), index=False
    )

    print(f"[evaluate] Results saved to {save_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_multi_seed_results(
    all_results: list[dict[str, dict]],
    class_names: list[str],
) -> dict:
    """
    Aggregate metric results across multiple seeds/splits.

    Returns nested dict:
      {method: {metric: {"mean": X, "std": Y, "values": [...]},
                "per_class": {cls: {"mean": X, "std": Y}}}}
    """
    n = len(all_results)
    methods = list(all_results[0].keys())
    agg: dict = {}

    for method in methods:
        scalar_keys = [k for k in all_results[0][method] if k != "per_class"]
        method_agg: dict = {}

        for key in scalar_keys:
            vals = [r[method][key] for r in all_results]
            method_agg[key] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals, ddof=1 if n > 1 else 0)),
                "values": [float(v) for v in vals],
            }

        per_class_agg: dict = {}
        for cls in class_names:
            cls_vals = [
                r[method]["per_class"].get(cls)
                for r in all_results
                if r[method].get("per_class", {}).get(cls) is not None
            ]
            if cls_vals:
                per_class_agg[cls] = {
                    "mean": float(np.mean(cls_vals)),
                    "std":  float(np.std(cls_vals, ddof=1 if len(cls_vals) > 1 else 0)),
                }
            else:
                per_class_agg[cls] = {"mean": None, "std": None}
        method_agg["per_class"] = per_class_agg

        agg[method] = method_agg

    return agg


def print_multi_seed_summary(agg: dict, n_seeds: int = 5) -> None:
    """Print aggregated multi-seed results."""
    methods = [k for k in agg if not k.startswith("_")]
    print("\n" + "=" * 92)
    print(f"MULTI-SEED SUMMARY  (mean ± std  across {n_seeds} random splits)")
    print("=" * 92)
    header = (
        f"{'Method':<28} {'Top-1':>17} {'GMeans':>17}"
        f" {'bAcc':>17} {'MacroF1':>17}"
    )
    print(header)
    print("-" * 92)
    for method in methods:
        m = agg[method]

        def fmt(key: str) -> str:
            if key in m:
                return f"{m[key]['mean']:.4f} ± {m[key]['std']:.4f}"
            return "      N/A      "

        print(
            f"{method:<28} {fmt('top1'):>17} {fmt('g_means'):>17}"
            f" {fmt('bacc'):>17} {fmt('macro_f1'):>17}"
        )
    print("=" * 92)


def save_multi_seed_results(
    agg: dict,
    save_dir: str,
    extra: dict | None = None,
) -> None:
    """Save aggregated multi-seed metrics to JSON and per-class CSV."""
    os.makedirs(save_dir, exist_ok=True)

    methods = [k for k in agg if not k.startswith("_")]

    # Scalar metrics → JSON
    summary: dict = {}
    for method in methods:
        summary[method] = {
            k: v for k, v in agg[method].items() if k != "per_class"
        }
    if extra:
        summary["_meta"] = extra
    with open(os.path.join(save_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Per-class accuracy → CSV
    rows = []
    for method in methods:
        for cls, stats in agg[method].get("per_class", {}).items():
            rows.append({
                "method":        method,
                "class":         cls,
                "mean_accuracy": stats["mean"],
                "std_accuracy":  stats["std"],
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, "per_class_summary.csv"), index=False
    )

    print(f"[evaluate] Multi-seed results saved to {save_dir}")
