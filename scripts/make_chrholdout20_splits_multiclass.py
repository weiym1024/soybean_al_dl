#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WINDOW_TABLE = ROOT / "data" / "al_shared_bg_peak_windows.tsv"
OUTROOT = ROOT / "results" / "al_shared_bg_multiclass_chrholdout20"
OUTROOT.mkdir(parents=True, exist_ok=True)

CHROMS = [f"Gm{str(i).zfill(2)}" for i in range(1, 21)]


def main():
    df = pd.read_csv(WINDOW_TABLE, sep="\t")

    required = {"peak_id", "chrom", "label", "label_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # peak-level unique table
    peak_df = (
        df[["peak_id", "chrom", "label", "label_id"]]
        .drop_duplicates()
        .copy()
    )

    # sanity: one peak_id should map to one chrom and one label
    chk = peak_df.groupby("peak_id").agg(
        n_chrom=("chrom", "nunique"),
        n_label=("label_id", "nunique"),
    )
    bad = chk[(chk["n_chrom"] > 1) | (chk["n_label"] > 1)]
    if len(bad) > 0:
        raise ValueError(
            f"Found {len(bad)} peak_id(s) mapping to multiple chroms or labels. Example: {bad.head().to_dict()}"
        )

    peak_df = peak_df.drop_duplicates(subset=["peak_id"]).reset_index(drop=True)

    all_chroms_in_data = sorted(peak_df["chrom"].dropna().unique().tolist())
    print("[INFO] Chromosomes in data:", all_chroms_in_data)
    print("[INFO] Total peaks:", len(peak_df))

    all_rows = []

    for i, test_chr in enumerate(CHROMS):
        val_chr = CHROMS[(i + 1) % len(CHROMS)]
        train_chrs = [c for c in CHROMS if c not in {test_chr, val_chr}]

        fold_name = f"fold_{i+1:02d}_test_{test_chr}_val_{val_chr}"
        outdir = OUTROOT / fold_name
        outdir.mkdir(parents=True, exist_ok=True)

        train_peaks = peak_df[peak_df["chrom"].isin(train_chrs)].copy()
        val_peaks = peak_df[peak_df["chrom"] == val_chr].copy()
        test_peaks = peak_df[peak_df["chrom"] == test_chr].copy()

        train_peaks.to_csv(outdir / "train_peaks.tsv", sep="\t", index=False)
        val_peaks.to_csv(outdir / "val_peaks.tsv", sep="\t", index=False)
        test_peaks.to_csv(outdir / "test_peaks.tsv", sep="\t", index=False)

        summary_rows = []
        for split_name, sub in [
            ("train", train_peaks),
            ("val", val_peaks),
            ("test", test_peaks),
        ]:
            counts = (
                sub.groupby(["label_id", "label"])
                .size()
                .reset_index(name="n_peaks")
                .sort_values("label_id")
            )
            counts.insert(0, "split", split_name)
            counts.insert(0, "val_chr", val_chr)
            counts.insert(0, "test_chr", test_chr)
            counts.insert(0, "fold", fold_name)
            summary_rows.append(counts)

        summary_df = pd.concat(summary_rows, axis=0, ignore_index=True)
        summary_df.to_csv(outdir / "split_summary.tsv", sep="\t", index=False)

        all_rows.append(summary_df)

        print(
            f"[INFO] {fold_name}: "
            f"train={len(train_peaks)} peaks, val={len(val_peaks)} peaks, test={len(test_peaks)} peaks"
        )

    all_summary = pd.concat(all_rows, axis=0, ignore_index=True)
    all_summary.to_csv(OUTROOT / "all_folds_summary.tsv", sep="\t", index=False)
    print("[INFO] Saved all summary to:", OUTROOT / "all_folds_summary.tsv")


if __name__ == "__main__":
    main()
