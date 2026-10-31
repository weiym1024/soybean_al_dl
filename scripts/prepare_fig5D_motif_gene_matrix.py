#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare Fig.5D motif-gene matrix from prob060 Al-specific results."
    )
    p.add_argument("--peak-table", required=True,
                   help="TSV: prob060 Al top1-per-peak table with gene/RNA annotations")
    p.add_argument("--fimo", required=True,
                   help="FIMO TSV for Al-specific sequences")
    p.add_argument("--motif-summary", required=True,
                   help="STREME+Tomtom summary TSV")
    p.add_argument("--attr-dir", required=False, default=None,
                   help="Directory containing *.actual_contrib.npy files")
    p.add_argument("--top-motifs", type=int, default=10,
                   help="Number of top STREME motifs to retain")
    p.add_argument("--top-rows", type=int, default=12,
                   help="Number of representative peak/gene rows to plot")
    p.add_argument("--must-include-peak", default="Al_peak_004707",
                   help="Peak that must be included")
    p.add_argument("--must-include-gene", default=None,
                   help="Gene symbol / gene_id_short that must be included if present")
    p.add_argument("--outdir", required=True,
                   help="Output directory")
    return p.parse_args()


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_actual_contrib(attr_path: Path):
    arr = np.load(attr_path)
    # 支持 (4, L) 或 (L, 4)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected attr shape: {arr.shape} for {attr_path}")
    if arr.shape[0] == 4:
        pos_abs = np.abs(arr).sum(axis=0)   # (L,)
    elif arr.shape[1] == 4:
        pos_abs = np.abs(arr).sum(axis=1)   # (L,)
    else:
        raise ValueError(f"Unexpected attr shape: {arr.shape} for {attr_path}")
    return pos_abs


def choose_gene_id_col(df):
    for c in ["gene_id_short", "gene_symbol", "gene_id_full", "gene_id", "promoter_id"]:
        if c in df.columns:
            return c
    return None


def choose_log2fc_col(df):
    for c in ["log2FoldChange", "log2FC", "lfc"]:
        if c in df.columns:
            return c
    return None


def choose_padj_col(df):
    for c in ["padj", "FDR", "adj_pvalue"]:
        if c in df.columns:
            return c
    return None


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1. peak table
    # -------------------------
    peaks = pd.read_csv(args.peak_table, sep="\t")
    print(f"[INFO] Loaded peak table: {peaks.shape[0]} rows")

    gene_col = choose_gene_id_col(peaks)
    lfc_col = choose_log2fc_col(peaks)
    padj_col = choose_padj_col(peaks)

    if gene_col is None:
        peaks["gene_id_plot"] = peaks["peak_id"].astype(str)
    else:
        peaks["gene_id_plot"] = peaks[gene_col].fillna("").astype(str)
        peaks.loc[peaks["gene_id_plot"].eq(""), "gene_id_plot"] = peaks["peak_id"].astype(str)

    if "prob_2" not in peaks.columns:
        raise ValueError("[ERROR] peak table must contain prob_2")

    if "window_id" not in peaks.columns or "peak_id" not in peaks.columns:
        raise ValueError("[ERROR] peak table must contain window_id and peak_id")

    peaks["prob_2"] = peaks["prob_2"].astype(float)
    if lfc_col is not None:
        peaks[lfc_col] = pd.to_numeric(peaks[lfc_col], errors="coerce")
    else:
        peaks["log2FoldChange"] = np.nan
        lfc_col = "log2FoldChange"

    if padj_col is not None:
        peaks[padj_col] = pd.to_numeric(peaks[padj_col], errors="coerce")
    else:
        peaks["padj"] = np.nan
        padj_col = "padj"

    peaks["display_name"] = peaks["gene_id_plot"].astype(str) + " | " + peaks["peak_id"].astype(str)

    # -------------------------
    # 2. motif summary -> top motifs
    # -------------------------
    ms = pd.read_csv(args.motif_summary, sep="\t")
    print(f"[INFO] Loaded motif summary: {ms.shape[0]} rows")

    # 优先按 tomtom_qvalue, 然后 streme_pvalue
    if "tomtom_qvalue" in ms.columns:
        ms["tomtom_qvalue"] = pd.to_numeric(ms["tomtom_qvalue"], errors="coerce")
    else:
        ms["tomtom_qvalue"] = np.nan
    if "streme_motif_pvalue" in ms.columns:
        ms["streme_motif_pvalue"] = pd.to_numeric(ms["streme_motif_pvalue"], errors="coerce")
    else:
        ms["streme_motif_pvalue"] = np.nan

    ms = ms.sort_values(
        by=["tomtom_qvalue", "streme_motif_pvalue", "n_positive_matches"],
        ascending=[True, True, False],
        na_position="last"
    ).reset_index(drop=True)

    top_ms = ms.head(args.top_motifs).copy()
    top_motif_ids = top_ms["motif_ID"].tolist()

    # 给列标签一个更好看的名字
    def motif_label(row):
        streme = str(row["motif_ALT_ID"]) if "motif_ALT_ID" in row else str(row["motif_ID"])
        jaspar = ""
        if "jaspar_target_id" in row and pd.notna(row["jaspar_target_id"]):
            jaspar = str(row["jaspar_target_id"])
        consensus = ""
        if "consensus" in row and pd.notna(row["consensus"]):
            consensus = str(row["consensus"])
        if jaspar:
            return f"{streme}\n{jaspar}\n{consensus}"
        return f"{streme}\n{consensus}"

    top_ms["motif_label"] = top_ms.apply(motif_label, axis=1)
    top_ms.to_csv(outdir / "fig5D_top10_motifs.tsv", sep="\t", index=False)
    print(f"[INFO] Saved top motif table: {outdir / 'fig5D_top10_motifs.tsv'}")

    # -------------------------
    # 3. fimo hits
    # -------------------------
    fimo = pd.read_csv(args.fimo, sep="\t", comment="#")
    fimo.columns = [c.strip() for c in fimo.columns]
    print(f"[INFO] Loaded FIMO: {fimo.shape[0]} rows")

    rename_map = {}
    if "sequence_name" in fimo.columns:
        rename_map["sequence_name"] = "seq_name"
    elif "sequence_name " in fimo.columns:
        rename_map["sequence_name "] = "seq_name"
    elif "sequence" in fimo.columns:
        rename_map["sequence"] = "seq_name"
    elif "sequence_id" in fimo.columns:
        rename_map["sequence_id"] = "seq_name"

    if "motif_id" in fimo.columns:
        rename_map["motif_id"] = "motif_ID"
    if "motif_alt_id" in fimo.columns:
        rename_map["motif_alt_id"] = "motif_ALT_ID"
    if "start" in fimo.columns:
        rename_map["start"] = "motif_start"
    if "stop" in fimo.columns:
        rename_map["stop"] = "motif_stop"
    if "q-value" in fimo.columns:
        rename_map["q-value"] = "qvalue"
    if "p-value" in fimo.columns:
        rename_map["p-value"] = "pvalue"
    if "matched_sequence" in fimo.columns:
        rename_map["matched_sequence"] = "matched_sequence"

    fimo = fimo.rename(columns=rename_map)

    required_cols = ["motif_ID", "seq_name", "motif_start", "motif_stop"]
    for c in required_cols:
        if c not in fimo.columns:
            raise ValueError(f"[ERROR] FIMO missing required column: {c}")

    fimo = fimo[fimo["motif_ID"].isin(top_motif_ids)].copy()
    fimo["window_id"] = fimo["seq_name"].astype(str).str.split("|").str[0]
    fimo["motif_start"] = pd.to_numeric(fimo["motif_start"], errors="coerce")
    fimo["motif_stop"] = pd.to_numeric(fimo["motif_stop"], errors="coerce")
    if "qvalue" in fimo.columns:
        fimo["qvalue"] = pd.to_numeric(fimo["qvalue"], errors="coerce")
    else:
        fimo["qvalue"] = np.nan
    if "pvalue" in fimo.columns:
        fimo["pvalue"] = pd.to_numeric(fimo["pvalue"], errors="coerce")
    else:
        fimo["pvalue"] = np.nan

    # 合并 peak table
    merged = fimo.merge(
        peaks,
        on="window_id",
        how="inner",
        suffixes=("", "_peak")
    )
    print(f"[INFO] FIMO hits after merge with peaks: {merged.shape[0]} rows")

    # -------------------------
    # 4. add motif attribution (if attr-dir provided)
    # -------------------------
    merged["motif_attr"] = np.nan
    if args.attr_dir is not None:
        attr_dir = Path(args.attr_dir)
        cache = {}
        n_ok = 0
        for idx, row in merged.iterrows():
            wid = row["window_id"]
            if wid not in cache:
                attr_path = attr_dir / f"{wid}.actual_contrib.npy"
                if not attr_path.exists():
                    cache[wid] = None
                else:
                    try:
                        cache[wid] = load_actual_contrib(attr_path)
                    except Exception:
                        cache[wid] = None
            pos_abs = cache[wid]
            if pos_abs is None:
                continue

            # 如果 motif_start/stop 是 genomic coord，则转成相对窗口位置
            # 否则如果本来就是 1-based local coord，也兼容
            wstart = row["start"]
            wend = row["end"]
            mstart = int(row["motif_start"])
            mstop = int(row["motif_stop"])

            if mstart >= int(wstart) and mstop <= int(wend):
                # genomic coords -> local coords
                local_start = mstart - int(wstart) + 1
                local_stop = mstop - int(wstart) + 1
            else:
                # assume already local 1-based
                local_start = mstart
                local_stop = mstop

            local_start = max(1, local_start)
            local_stop = min(len(pos_abs), local_stop)
            if local_start <= local_stop:
                val = float(np.mean(pos_abs[local_start - 1: local_stop]))
                merged.at[idx, "motif_attr"] = val
                n_ok += 1

        print(f"[INFO] motif_attr assigned for {n_ok} motif hits")

    # -------------------------
    # 5. aggregate motif × peak/gene
    # -------------------------
    agg_cols = {
        "qvalue": "min",
        "pvalue": "min",
        "motif_attr": "mean",
        "matched_sequence": lambda x: ";".join(sorted(set(map(str, x.dropna().tolist()))))[:500]
    }

    group_cols = [
        "window_id", "peak_id", "chrom", "start", "end",
        "gene_id_plot", "display_name", "prob_2", lfc_col, padj_col,
        "motif_ID"
    ]

    agg = merged.groupby(group_cols, as_index=False).agg(agg_cols)
    nh = merged.groupby(group_cols, as_index=False).size().rename(columns={"size": "n_hits"})
    agg = agg.merge(nh, on=group_cols, how="left")

    agg = agg.merge(
        top_ms[["motif_ID", "motif_ALT_ID", "consensus", "jaspar_target_id", "motif_label"]],
        on="motif_ID",
        how="left"
    )

    agg = agg.rename(columns={
        lfc_col: "log2FoldChange",
        padj_col: "padj"
    })

    agg.to_csv(outdir / "fig5D_motif_gene_long_all.tsv", sep="\t", index=False)
    print(f"[INFO] Saved all long table: {outdir / 'fig5D_motif_gene_long_all.tsv'}")

    # -------------------------
    # 6. select representative rows
    # -------------------------
    row_stat = agg.groupby(
        ["window_id", "peak_id", "chrom", "start", "end", "gene_id_plot", "display_name", "prob_2", "log2FoldChange", "padj"],
        as_index=False
    ).agg(
        n_top_motifs=("motif_ID", "nunique"),
        total_hits=("n_hits", "sum"),
        mean_motif_attr=("motif_attr", "mean"),
        best_motif_attr=("motif_attr", "max"),
        best_qvalue=("qvalue", "min")
    )

    row_stat["best_qvalue_log10"] = -np.log10(row_stat["best_qvalue"].clip(lower=1e-300))
    row_stat["log2FoldChange_pos"] = row_stat["log2FoldChange"].fillna(0).clip(lower=0)
    row_stat["score"] = (
        row_stat["prob_2"].fillna(0) * 3.0 +
        row_stat["log2FoldChange_pos"] * 1.0 +
        row_stat["n_top_motifs"].fillna(0) * 0.4 +
        row_stat["best_qvalue_log10"].fillna(0) * 0.15 +
        row_stat["best_motif_attr"].fillna(0) * 200.0
    )

    row_stat = row_stat.sort_values(
        by=["score", "prob_2", "log2FoldChange", "n_top_motifs"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    selected = []

    # 必须包含 peak
    if args.must_include_peak is not None:
        sub = row_stat[row_stat["peak_id"].astype(str) == str(args.must_include_peak)]
        if sub.shape[0] > 0:
            selected.append(sub.iloc[0])

    # 必须包含 gene
    if args.must_include_gene is not None:
        sub = row_stat[row_stat["gene_id_plot"].astype(str).str.contains(str(args.must_include_gene), na=False)]
        if sub.shape[0] > 0:
            selected.append(sub.iloc[0])

    used = set()
    sel_rows = []
    for s in selected:
        key = s["window_id"]
        if key not in used:
            sel_rows.append(s)
            used.add(key)

    for _, r in row_stat.iterrows():
        if len(sel_rows) >= args.top_rows:
            break
        if r["window_id"] in used:
            continue
        sel_rows.append(r)
        used.add(r["window_id"])

    sel_df = pd.DataFrame(sel_rows).reset_index(drop=True)
    sel_df.to_csv(outdir / "fig5D_selected_rows.tsv", sep="\t", index=False)
    print(f"[INFO] Saved selected rows: {outdir / 'fig5D_selected_rows.tsv'}")

    # 只保留 selected 行
    keep_wids = set(sel_df["window_id"].tolist())
    plot_long = agg[agg["window_id"].isin(keep_wids)].copy()

    # 保证 top motif 也齐全
    plot_long.to_csv(outdir / "fig5D_motif_gene_long_selected.tsv", sep="\t", index=False)
    print(f"[INFO] Saved plot long table: {outdir / 'fig5D_motif_gene_long_selected.tsv'}")

    # 额外输出一个 presence matrix 便于检查
    pres = plot_long.pivot_table(
        index="display_name",
        columns="motif_ID",
        values="n_hits",
        aggfunc="sum",
        fill_value=0
    )
    pres.to_csv(outdir / "fig5D_presence_matrix.tsv", sep="\t")
    print(f"[INFO] Saved presence matrix: {outdir / 'fig5D_presence_matrix.tsv'}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
