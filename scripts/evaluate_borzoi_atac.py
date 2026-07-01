#!/usr/bin/env python
"""Evaluate a fine-tuned Borzoi ATAC model on fold_1-style BED regions.

This mirrors the metric outputs from alphagenome-pytorch's
evaluate_checkpoint.py and evaluate_checkpoint_borzoi32.py:
  - 32bp metrics compare Borzoi 32bp predictions to 32bp sum-binned BigWig.
  - 1bp metrics repeat each 32bp Borzoi prediction 32 times and compare to
    raw 1bp BigWig signal over the same centered Borzoi target window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import pysam
from scipy import stats
import tensorflow as tf
from tqdm import tqdm

from baskerville import dna, seqnn


def compute_all_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    n_regions = preds.shape[0]
    profile_rs = []
    for i in range(n_regions):
        p = preds[i].flatten()
        t = targets[i].flatten()
        if np.std(t) > 1e-10 and np.std(p) > 1e-10:
            profile_rs.append(stats.pearsonr(p, t)[0])
        else:
            profile_rs.append(0.0)
    profile_rs = np.asarray(profile_rs)

    pred_counts = preds.sum(axis=1).flatten()
    target_counts = targets.sum(axis=1).flatten()
    if np.std(pred_counts) > 1e-10 and np.std(target_counts) > 1e-10:
        count_r = stats.pearsonr(pred_counts, target_counts)[0]
    else:
        count_r = 0.0

    eps = 1e-8
    p = targets / (targets.sum(axis=1, keepdims=True) + eps)
    q = preds / (preds.sum(axis=1, keepdims=True) + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)), axis=1)
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)), axis=1)
    jsd_per_reg = (0.5 * (kl_pm + kl_qm)).mean(axis=1)

    p_flat = preds.flatten()
    t_flat = targets.flatten()
    if len(p_flat) > 2_000_000:
        idx = np.random.default_rng(42).choice(len(p_flat), 2_000_000, replace=False)
        p_flat = p_flat[idx]
        t_flat = t_flat[idx]

    return {
        "profile_pearson_r_all": profile_rs,
        "profile_pearson_r_mean": float(np.mean(profile_rs)),
        "profile_pearson_r_median": float(np.median(profile_rs)),
        "count_pearson_r": float(count_r),
        "jsd_all": jsd_per_reg,
        "jsd_mean": float(np.mean(jsd_per_reg)),
        "jsd_median": float(np.median(jsd_per_reg)),
        "mse": float(np.mean((preds - targets) ** 2)),
        "spearman_global": float(stats.spearmanr(p_flat, t_flat)[0]),
        "n_regions": n_regions,
    }


def plot_scatter(preds: np.ndarray, targets: np.ndarray, out_path: Path, title_suffix: str) -> None:
    p = preds.flatten()
    t = targets.flatten()
    n = min(len(p), 100_000)
    idx = np.random.default_rng(42).choice(len(p), n, replace=False)
    p = p[idx]
    t = t[idx]
    r = stats.pearsonr(p, t)[0] if np.std(p) > 1e-10 and np.std(t) > 1e-10 else 0.0
    lim = max(float(np.nanmax(t)), float(np.nanmax(p)), 1e-6) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.05, s=1, color="steelblue", rasterized=True)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Observed signal")
    ax.set_ylabel("Predicted signal")
    ax.set_title(f"Pred vs Obs {title_suffix} (r={r:.3f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_counts(preds: np.ndarray, targets: np.ndarray, out_path: Path, title_suffix: str) -> None:
    p = preds.sum(axis=1).flatten()
    t = targets.sum(axis=1).flatten()
    r = stats.pearsonr(p, t)[0] if np.std(p) > 1e-10 and np.std(t) > 1e-10 else 0.0
    lim = max(float(np.nanmax(t)), float(np.nanmax(p)), 1e-6) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, alpha=0.15, s=5, color="steelblue", rasterized=True)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Observed total count")
    ax.set_ylabel("Predicted total count")
    ax.set_title(f"Count correlation {title_suffix} (r={r:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hist(values: np.ndarray, out_path: Path, xlabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=50, alpha=0.6, color="steelblue", edgecolor="white")
    ax.axvline(np.median(values), color="steelblue", linestyle="--", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def read_bed(path: str) -> list[tuple[str, int, int]]:
    regions = []
    with open(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            chrom, start, end, *_ = line.split()
            regions.append((chrom, int(start), int(end)))
    return regions


def centered_window(chrom: str, start: int, end: int, length: int, chrom_sizes: dict[str, int]):
    center = (start + end) // 2
    seq_start = center - length // 2
    seq_end = seq_start + length
    if seq_start < 0 or seq_end > chrom_sizes.get(chrom, 0):
        return None
    return chrom, seq_start, seq_end


def bigwig_values(bw, chrom: str, start: int, end: int) -> np.ndarray:
    vals = np.asarray(bw.values(chrom, start, end, numpy=True), dtype=np.float32)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    return vals


def write_outputs(out_dir: Path, res: int, preds: np.ndarray, targets: np.ndarray, metrics: dict, meta: dict) -> None:
    metrics_dir = out_dir / "metrics"
    pred_dir = out_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    res_tag = f"{res}bp"
    plot_scatter(preds, targets, metrics_dir / f"scatter_test_{res_tag}.png", f"(test, {res_tag})")
    plot_scatter_counts(preds, targets, metrics_dir / f"scatter_counts_test_{res_tag}.png", f"(test, {res_tag})")
    plot_hist(metrics["profile_pearson_r_all"], metrics_dir / f"correlation_hist_test_{res_tag}.png", "Pearson r (per region)", f"Borzoi profile correlation distribution ({res_tag})")
    plot_hist(metrics["jsd_all"], metrics_dir / f"jsd_hist_test_{res_tag}.png", "JSD (per region)", f"Borzoi JSD distribution ({res_tag})")

    np.save(pred_dir / f"test_preds_{res_tag}.npy", preds.astype(np.float16))
    np.save(pred_dir / f"test_targets_{res_tag}.npy", targets.astype(np.float16))

    clean_metrics = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    summary = {
        **meta,
        "bin_size": res,
        "n_regions": metrics["n_regions"],
        "metrics": clean_metrics,
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    summary_text = (
        f"Borzoi eval @ {res_tag}\n"
        f"{'-' * 60}\n"
        f"Profile r (mean):   {metrics['profile_pearson_r_mean']:.4f}\n"
        f"Profile r (median): {metrics['profile_pearson_r_median']:.4f}\n"
        f"Count r:            {metrics['count_pearson_r']:.4f}\n"
        f"JSD (mean):         {metrics['jsd_mean']:.4f}\n"
        f"JSD (median):       {metrics['jsd_median']:.4f}\n"
        f"MSE:                {metrics['mse']:.4f}\n"
        f"Spearman (global):  {metrics['spearman_global']:.4f}\n"
        f"N regions:          {metrics['n_regions']}\n"
    )
    print(summary_text)
    with open(out_dir / "summary.txt", "w") as handle:
        handle.write(summary_text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--genome", required=True)
    p.add_argument("--bigwig", required=True)
    p.add_argument("--test-bed", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-regions", type=int, default=0)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument(
        "--inverse-sum-sqrt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert Baskerville sum_sqrt transform: raw_sum=(pred+1)^2-1.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_base = Path(args.output_dir)
    out_1bp = Path(str(out_base) + "_1bp")
    out_32bp = Path(str(out_base) + "_32bp")
    if (out_1bp / "summary.json").exists() and (out_32bp / "summary.json").exists():
        print(f"Evaluation already complete: {out_dir}")
        return

    with open(args.params) as handle:
        params = json.load(handle)
    params_model = params["model"]
    seq_length = int(params_model["seq_length"])
    pool_width = 32
    target_length = seq_length - 2 * 163840

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = seqnn.SeqNN(params_model)
    model.restore(args.model)
    if args.mixed_precision:
        model.append_activation()

    fasta = pysam.Fastafile(args.genome)
    chrom_sizes = {name: length for name, length in zip(fasta.references, fasta.lengths)}
    bw = pyBigWig.open(args.bigwig)

    regions = read_bed(args.test_bed)
    if args.max_regions and len(regions) > args.max_regions:
        idx = np.random.default_rng(42).choice(len(regions), args.max_regions, replace=False)
        regions = [regions[i] for i in idx]

    preds_32, targets_32, targets_1 = [], [], []
    batch = []
    batch_windows = []

    def flush_batch():
        if not batch:
            return
        x = np.stack(batch).astype(np.float32)
        y = model(x, dtype="float32")
        for yi, (chrom, target_start, target_end) in zip(y, batch_windows):
            pred32 = yi[:, 0].astype(np.float32)
            if args.inverse_sum_sqrt:
                pred32 = np.maximum((pred32 + 1.0) ** 2 - 1.0, 0.0)
            target1 = bigwig_values(bw, chrom, target_start, target_end)
            target32 = target1.reshape(-1, pool_width).sum(axis=1, dtype=np.float32)
            preds_32.append(pred32[:, None])
            targets_32.append(target32[:, None])
            targets_1.append(target1[:, None])
        batch.clear()
        batch_windows.clear()

    for chrom, start, end in tqdm(regions, desc="Borzoi eval"):
        seq_window = centered_window(chrom, start, end, seq_length, chrom_sizes)
        target_window = centered_window(chrom, start, end, target_length, chrom_sizes)
        if seq_window is None or target_window is None:
            continue
        _, seq_start, seq_end = seq_window
        _, target_start, target_end = target_window
        seq = fasta.fetch(chrom, seq_start, seq_end)
        batch.append(dna.dna_1hot(seq, seq_len=seq_length).astype(np.float32))
        batch_windows.append((chrom, target_start, target_end))
        if len(batch) >= args.batch_size:
            flush_batch()
    flush_batch()

    fasta.close()
    bw.close()

    preds32 = np.stack(preds_32)
    targets32 = np.stack(targets_32)
    targets1 = np.stack(targets_1)
    preds1 = np.repeat(preds32, pool_width, axis=1)

    meta = {
        "checkpoint": args.model,
        "params": args.params,
        "genome": args.genome,
        "bigwig": args.bigwig,
        "test_bed": args.test_bed,
        "sequence_length": seq_length,
        "target_length": target_length,
        "prediction_note": "1bp predictions are 32bp Borzoi predictions repeated 32 times.",
        "inverse_sum_sqrt": args.inverse_sum_sqrt,
    }

    write_outputs(out_32bp, 32, preds32, targets32, compute_all_metrics(preds32, targets32), meta)
    write_outputs(out_1bp, 1, preds1, targets1, compute_all_metrics(preds1, targets1), meta)


if __name__ == "__main__":
    main()
