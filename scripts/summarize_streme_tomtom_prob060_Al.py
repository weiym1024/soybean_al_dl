#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MOTIF_DIR = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42" / "prob060_Al_attr" / "motif_discovery"

STREME_TXT = MOTIF_DIR / "streme_out" / "streme.txt"
SITES = MOTIF_DIR / "streme_out" / "sites.tsv"
SEQUENCES = MOTIF_DIR / "streme_out" / "sequences.tsv"
TOMTOM_BEST = MOTIF_DIR / "tomtom_JASPAR_besthit_per_query.tsv"

OUT = MOTIF_DIR / "streme_tomtom_summary.tsv"


def parse_streme_motifs(streme_txt: Path):
    rows = []
    with open(streme_txt) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF "):
                parts = line.split()
                # example: MOTIF 1-GACGVTGSW STREME-1
                motif_id = parts[1]
                alt_id = parts[2] if len(parts) > 2 else ""
                consensus = motif_id.split("-", 1)[1] if "-" in motif_id else motif_id
                rows.append({
                    "motif_ID": motif_id,
                    "motif_ALT_ID": alt_id,
                    "consensus": consensus,
                    "width": len(consensus),
                })
    return pd.DataFrame(rows)


def main():
    motif_df = parse_streme_motifs(STREME_TXT)

    sites = pd.read_csv(SITES, sep="\t")
    seqs = pd.read_csv(SEQUENCES, sep="\t")

    # site-level count
    site_counts = (
        sites.groupby(["motif_ID", "motif_ALT_ID"])
        .size()
        .reset_index(name="n_sites")
    )

    # sequence-level tp/fp counts
    seqs["seq_Class"] = seqs["seq_Class"].astype(str)
    seq_counts = (
        seqs.groupby(["motif_ID", "motif_ALT_ID", "seq_Class"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["tp", "fp"]:
        if col not in seq_counts.columns:
            seq_counts[col] = 0

    seq_counts = seq_counts.rename(columns={
        "tp": "n_positive_matches",
        "fp": "n_background_matches",
    })

    # motif p-value
    pvals = (
        seqs.groupby(["motif_ID", "motif_ALT_ID"])["motif_P-value"]
        .first()
        .reset_index()
        .rename(columns={"motif_P-value": "streme_motif_pvalue"})
    )

    out = motif_df.merge(site_counts, on=["motif_ID", "motif_ALT_ID"], how="left")
    out = out.merge(seq_counts[["motif_ID", "motif_ALT_ID", "n_positive_matches", "n_background_matches"]],
                    on=["motif_ID", "motif_ALT_ID"], how="left")
    out = out.merge(pvals, on=["motif_ID", "motif_ALT_ID"], how="left")

    if TOMTOM_BEST.exists():
        tom = pd.read_csv(TOMTOM_BEST, sep="\t")
        tom = tom.rename(columns={
            "Query_ID": "motif_ID",
            "Target_ID": "jaspar_target_id",
            "p-value": "tomtom_pvalue",
            "E-value": "tomtom_evalue",
            "q-value": "tomtom_qvalue",
            "Target_consensus": "jaspar_target_consensus",
            "Orientation": "tomtom_orientation",
            "Overlap": "tomtom_overlap",
        })
        keep_cols = [
            "motif_ID",
            "jaspar_target_id",
            "tomtom_pvalue",
            "tomtom_evalue",
            "tomtom_qvalue",
            "tomtom_overlap",
            "jaspar_target_consensus",
            "tomtom_orientation",
        ]
        keep_cols = [c for c in keep_cols if c in tom.columns]
        out = out.merge(tom[keep_cols], on="motif_ID", how="left")

    out = out.sort_values(["streme_motif_pvalue", "tomtom_qvalue"], ascending=[True, True])
    out.to_csv(OUT, sep="\t", index=False)

    print("[INFO] Saved:", OUT)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
