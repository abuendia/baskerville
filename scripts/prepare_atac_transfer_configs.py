#!/usr/bin/env python3
"""Prepare one-track Borzoi transfer targets and params files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRUNK = [
    {
        "name": "conv_dna",
        "filters": 512,
        "kernel_size": 15,
        "norm_type": None,
        "activation": "linear",
        "pool_size": 2,
    },
    {
        "name": "res_tower",
        "filters_init": 608,
        "filters_end": 1536,
        "divisible_by": 32,
        "kernel_size": 5,
        "num_convs": 1,
        "pool_size": 2,
        "repeat": 6,
    },
    {
        "name": "transformer_tower",
        "key_size": 64,
        "heads": 8,
        "num_position_features": 32,
        "dropout": 0.2,
        "mha_l2_scale": 1.0e-8,
        "l2_scale": 1.0e-8,
        "kernel_initializer": "he_normal",
        "repeat": 8,
    },
    {"name": "unet_conv", "kernel_size": 3, "upsample_conv": True},
    {"name": "unet_conv", "kernel_size": 3, "upsample_conv": True},
    {"name": "Cropping1D", "cropping": 5120},
    {"name": "conv_nac", "filters": 1920, "dropout": 0.1},
]


MODE_CONFIGS = {
    "full": {
        "learning_rate": 0.000006,
        "transfer": {"mode": "full"},
    },
    "linear": {
        "learning_rate": 0.00006,
        "transfer": {"mode": "linear"},
    },
    "lora": {
        "learning_rate": 0.00006,
        "transfer": {
            "mode": "adapter",
            "adapter": "lora",
            "adapter_latent": 8,
            "lora_alpha": 16,
        },
    },
    "locon": {
        "learning_rate": 0.00006,
        "transfer": {
            "mode": "adapter",
            "adapter": "locon",
            "conv_select": 4,
            "conv_latent": 4,
            "locon_alpha": 1,
        },
    },
}


def make_params(args: argparse.Namespace, mode: str) -> dict:
    mode_cfg = MODE_CONFIGS[mode]
    return {
        "train": {
            "batch_size": args.batch_size,
            "shuffle_buffer": args.shuffle_buffer,
            "optimizer": "adam",
            "learning_rate": mode_cfg["learning_rate"],
            "loss": "poisson_mn",
            "total_weight": args.total_weight,
            "warmup_steps": args.warmup_steps,
            "global_clipnorm": args.global_clipnorm,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "patience": args.patience,
            "train_epochs_min": args.epochs_min,
            "train_epochs_max": args.epochs_max,
        },
        "transfer": mode_cfg["transfer"],
        "model": {
            "seq_length": args.seq_length,
            "augment_rc": True,
            "augment_shift": 3,
            "activation": "gelu",
            "norm_type": "batch-sync",
            "bn_momentum": 0.9,
            "kernel_initializer": "lecun_normal",
            "l2_scale": 2.0e-8,
            "trunk": TRUNK,
            "head_human": {
                "name": "final",
                "units": 1,
                "activation": "softplus",
            },
        },
    }


def write_targets(args: argparse.Namespace) -> None:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.data_dir / "targets.txt"
    with out_path.open("w") as out:
        out.write(
            "\tidentifier\tfile\tclip\tclip_soft\tscale\tsum_stat\t"
            "strand_pair\tdescription\n"
        )
        out.write(
            f"0\t{args.identifier}\t{args.w5}\t{args.clip}\t"
            f"{args.clip_soft}\t{args.scale}\t{args.sum_stat}\t0\t"
            f"{args.description}\n"
        )


def read_fai(fasta: Path) -> dict[str, int]:
    fai = Path(str(fasta) + ".fai")
    if not fai.exists():
        raise FileNotFoundError(f"Missing FASTA index: {fai}")
    chrom_sizes = {}
    with fai.open() as handle:
        for line in handle:
            fields = line.split("\t")
            chrom_sizes[fields[0]] = int(fields[1])
    return chrom_sizes


def write_ag_fold_sequences(args: argparse.Namespace) -> None:
    if args.ag_fold_dir is None:
        return
    if args.fasta is None:
        raise ValueError("--fasta is required with --ag-fold-dir")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    chrom_sizes = read_fai(args.fasta)
    target_length = args.seq_length - 2 * args.crop_bp
    if target_length <= 0:
        raise ValueError(
            f"seq_length - 2 * crop_bp must be positive; got {target_length}"
        )

    split_files = [
        ("train", args.ag_fold_dir / "train.bed"),
        ("valid", args.ag_fold_dir / "valid.bed"),
        ("test", args.ag_fold_dir / "test.bed"),
    ]
    counts = {}
    skipped = {}
    out_path = args.data_dir / "sequences.bed"
    with out_path.open("w") as out:
        for label, bed_path in split_files:
            counts[label] = 0
            skipped[label] = 0
            with bed_path.open() as bed:
                for line in bed:
                    if not line.strip() or line.startswith("#"):
                        continue
                    chrom, raw_start, raw_end, *_ = line.split()
                    start, end = int(raw_start), int(raw_end)
                    if chrom not in chrom_sizes:
                        skipped[label] += 1
                        continue
                    center = (start + end) // 2
                    seq_start = center - target_length // 2
                    seq_end = seq_start + target_length
                    if seq_start < 0 or seq_end > chrom_sizes[chrom]:
                        skipped[label] += 1
                        continue
                    out.write(f"{chrom}\t{seq_start}\t{seq_end}\t{label}\n")
                    counts[label] += 1

    print(f"Wrote AG fold sequences: {out_path}")
    for label in ("train", "valid", "test"):
        print(f"  {label}: {counts[label]} kept, {skipped[label]} skipped")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--params-dir", type=Path, required=True)
    parser.add_argument("--identifier", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--w5", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, default=524288)
    parser.add_argument("--crop-bp", type=int, default=163840)
    parser.add_argument("--ag-fold-dir", type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs-min", type=int, default=1)
    parser.add_argument("--epochs-max", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--total-weight", type=float, default=0.2)
    parser.add_argument("--global-clipnorm", type=float, default=0.15)
    parser.add_argument("--clip", type=float, default=768.0)
    parser.add_argument("--clip-soft", type=float, default=384.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--sum-stat", default="sum_sqrt")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["full", "linear", "lora", "locon"],
        choices=sorted(MODE_CONFIGS),
    )
    args = parser.parse_args()

    write_targets(args)
    write_ag_fold_sequences(args)
    args.params_dir.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        with (args.params_dir / f"borzoi_{mode}.json").open("w") as out:
            json.dump(make_params(args, mode), out, indent=4)
            out.write("\n")


if __name__ == "__main__":
    main()
