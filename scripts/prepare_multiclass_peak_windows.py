#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "al_shared_bg_peak_metadata.tsv"
GENOME_FA = ROOT / "input" / "genome.fa"
GENOME_FAI = ROOT / "input" / "genome.fa.fai"

OUT_TSV = ROOT / "data" / "al_shared_bg_peak_windows.tsv"
OUT_FA = ROOT / "input" / "sequences" / "al_shared_bg_peak_windows.fa"

WINDOW = 500
STRIDE = 50
FLANK = 100


def read_fasta(path: Path):
    seqs = {}
    name = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf).upper()
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if name is not None:
            seqs[name] = "".join(buf).upper()
    return seqs


def read_fai(path: Path):
    chrlen = {}
    with open(path) as f:
        for line in f:
            arr = line.strip().split("\t")
            chrlen[arr[0]] = int(arr[1])
    return chrlen


def make_windows_for_region(row, chrlen):
    peak_id = row["peak_id"]
    chrom = row["chrom"]
    start = int(row["start"])
    end = int(row["end"])
    label = row["label"]
    label_id = int(row["label_id"])

    aug_start = max(0, start - FLANK)
    aug_end = min(chrlen[chrom], end + FLANK)
    aug_len = aug_end - aug_start

    rows = []

    # If augmented region is shorter than 500 bp, build one centered window
    # and pad later if it hits chromosome boundaries.
    if aug_len < WINDOW:
        center = (aug_start + aug_end) // 2
        win_start = center - WINDOW // 2
        win_end = win_start + WINDOW

        left_pad = 0
        right_pad = 0

        if win_start < 0:
            left_pad = -win_start
            win_start = 0
        if win_end > chrlen[chrom]:
            right_pad = win_end - chrlen[chrom]
            win_end = chrlen[chrom]

        window_id = f"{peak_id}__w0001"
        rows.append([
            window_id, peak_id, chrom, win_start, win_end,
            left_pad, right_pad, label, label_id
        ])

    # Otherwise, use strict sliding windows with stride 50 bp
    else:
        n = 0
        s = aug_start
        while s + WINDOW <= aug_end:
            n += 1
            win_start = s
            win_end = s + WINDOW
            window_id = f"{peak_id}__w{n:04d}"
            rows.append([
                window_id, peak_id, chrom, win_start, win_end,
                0, 0, label, label_id
            ])
            s += STRIDE

    return rows


def extract_seq_with_padding(genome, chrom, start, end, left_pad, right_pad, window_len=500):
    seq = genome[chrom][start:end]
    seq = ("N" * left_pad) + seq + ("N" * right_pad)

    if len(seq) < window_len:
        seq = seq + ("N" * (window_len - len(seq)))
    elif len(seq) > window_len:
        seq = seq[:window_len]

    return seq.upper()


def main():
    print("[INFO] Reading metadata...")
    meta = pd.read_csv(META, sep="\t")

    required_cols = {"peak_id", "chrom", "start", "end", "label", "label_id"}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {sorted(missing)}")

    print("[INFO] Reading genome and chromosome lengths...")
    genome = read_fasta(GENOME_FA)
    chrlen = read_fai(GENOME_FAI)

    print("[INFO] Building windows...")
    all_rows = []
    for _, row in meta.iterrows():
        all_rows.extend(make_windows_for_region(row, chrlen))

    win_df = pd.DataFrame(all_rows, columns=[
        "window_id", "peak_id", "chrom", "start", "end",
        "left_pad", "right_pad", "label", "label_id"
    ])

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FA.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Writing window table...")
    win_df.to_csv(OUT_TSV, sep="\t", index=False)

    print("[INFO] Writing FASTA...")
    with open(OUT_FA, "w") as out:
        for _, row in win_df.iterrows():
            seq = extract_seq_with_padding(
                genome=genome,
                chrom=row["chrom"],
                start=int(row["start"]),
                end=int(row["end"]),
                left_pad=int(row["left_pad"]),
                right_pad=int(row["right_pad"]),
                window_len=WINDOW
            )
            out.write(f">{row['window_id']}\n")
            out.write(seq + "\n")

    print("[INFO] Done.")
    print(f"[INFO] Window table: {OUT_TSV}")
    print(f"[INFO] FASTA: {OUT_FA}")
    print("[INFO] Region counts by label:")
    print(meta["label"].value_counts())
    print("[INFO] Window counts by label:")
    print(win_df["label"].value_counts())


if __name__ == "__main__":
    main()
