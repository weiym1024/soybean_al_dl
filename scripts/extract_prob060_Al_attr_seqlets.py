#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import random

ROOT = Path(__file__).resolve().parents[1]

FASTA = ROOT / "input" / "sequences" / "al_shared_bg_peak_windows.fa"
ATTR_DIR = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42" / "prob060_Al_attr"
SUMMARY = ATTR_DIR / "prob060_attr_summary.tsv"

OUTDIR = ATTR_DIR / "motif_discovery"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_POS_FA = OUTDIR / "prob060_Al_positive_seqlets_31bp.fa"
OUT_BG_FA = OUTDIR / "prob060_Al_background_seqlets_31bp.fa"
OUT_TSV = OUTDIR / "prob060_Al_positive_seqlets_31bp.tsv"

SEQLET_HALF = 15
SEQLET_LEN = 2 * SEQLET_HALF + 1
TOP_K_PER_WINDOW = 3
MIN_DISTANCE = 20
SEED = 42


def read_fasta(path):
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


def valid_seq(s):
    s = s.upper()
    return len(s) == SEQLET_LEN and set(s).issubset({"A", "C", "G", "T"})


def pick_top_positions(scores, top_k=3, min_distance=20):
    """
    scores: 1D actual contribution summed across true base channels.
    Only positive scores are used.
    """
    order = np.argsort(scores)[::-1]
    picked = []

    for pos0 in order:
        if scores[pos0] <= 0:
            break

        if pos0 < SEQLET_HALF or pos0 >= len(scores) - SEQLET_HALF:
            continue

        too_close = any(abs(pos0 - p) < min_distance for p in picked)
        if too_close:
            continue

        picked.append(pos0)
        if len(picked) >= top_k:
            break

    return picked


def random_background_positions(scores, n, excluded_positions):
    L = len(scores)
    candidates = [
        i for i in range(SEQLET_HALF, L - SEQLET_HALF)
        if all(abs(i - p) >= MIN_DISTANCE for p in excluded_positions)
    ]
    if len(candidates) == 0:
        return []
    if len(candidates) <= n:
        return candidates
    return random.sample(candidates, n)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    seqs = read_fasta(FASTA)
    summary = pd.read_csv(SUMMARY, sep="\t")

    pos_records = []
    bg_records = []

    for _, row in summary.iterrows():
        window_id = str(row["window_id"])
        peak_id = str(row["peak_id"])

        if window_id not in seqs:
            print(f"[WARN] {window_id} not found in FASTA, skipped.")
            continue

        seq = seqs[window_id]
        contrib_path = ATTR_DIR / f"{window_id}.actual_contrib.npy"
        if not contrib_path.exists():
            print(f"[WARN] Missing attribution: {contrib_path}")
            continue

        actual_contrib = np.load(contrib_path)  # 4 x 500
        scores = actual_contrib.sum(axis=0)

        picked = pick_top_positions(
            scores,
            top_k=TOP_K_PER_WINDOW,
            min_distance=MIN_DISTANCE
        )

        # positive seqlets
        for rank, pos0 in enumerate(picked, start=1):
            start0 = pos0 - SEQLET_HALF
            end0 = pos0 + SEQLET_HALF + 1
            subseq = seq[start0:end0].upper()
            if not valid_seq(subseq):
                continue

            rec_id = f"{window_id}|{peak_id}|rank{rank}|center{pos0+1}|score{scores[pos0]:.6g}"
            pos_records.append({
                "seqlet_id": rec_id,
                "window_id": window_id,
                "peak_id": peak_id,
                "center_position_1based": pos0 + 1,
                "seqlet_start_1based": start0 + 1,
                "seqlet_end_1based": end0,
                "score": float(scores[pos0]),
                "seq": subseq,
            })

        # background seqlets from same windows
        bg_positions = random_background_positions(scores, len(picked), picked)
        for j, pos0 in enumerate(bg_positions, start=1):
            start0 = pos0 - SEQLET_HALF
            end0 = pos0 + SEQLET_HALF + 1
            subseq = seq[start0:end0].upper()
            if not valid_seq(subseq):
                continue

            rec_id = f"{window_id}|{peak_id}|bg{j}|center{pos0+1}"
            bg_records.append({
                "seqlet_id": rec_id,
                "window_id": window_id,
                "peak_id": peak_id,
                "center_position_1based": pos0 + 1,
                "seqlet_start_1based": start0 + 1,
                "seqlet_end_1based": end0,
                "score": float(scores[pos0]),
                "seq": subseq,
            })

    pos_df = pd.DataFrame(pos_records)
    bg_df = pd.DataFrame(bg_records)

    pos_df.to_csv(OUT_TSV, sep="\t", index=False)

    with open(OUT_POS_FA, "w") as f:
        for _, r in pos_df.iterrows():
            f.write(f">{r['seqlet_id']}\n{r['seq']}\n")

    with open(OUT_BG_FA, "w") as f:
        for _, r in bg_df.iterrows():
            f.write(f">{r['seqlet_id']}\n{r['seq']}\n")

    print("[INFO] Positive seqlets:", len(pos_df))
    print("[INFO] Background seqlets:", len(bg_df))
    print("[INFO] Saved:", OUT_POS_FA)
    print("[INFO] Saved:", OUT_BG_FA)
    print("[INFO] Saved:", OUT_TSV)
    print(pos_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
