#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

FASTA = ROOT / "input" / "sequences" / "al_shared_bg_peak_windows.fa"
MODEL_PATH = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42" / "best_model_by_val_Al_specific_auPR.pt"
OUTDIR = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42" / "single_window_mutagenesis_formal"
OUTDIR.mkdir(parents=True, exist_ok=True)


WINDOW_IDS = [
    "Al_peak_004707__w0003",
]

SEQ_LEN = 500
CONV_OUT = 256
CONV_KERNEL = 12
N_CLASSES = 3
TARGET_CLASS = 2

BASES = ["A", "C", "G", "T"]


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


def one_hot(seq: str, seq_len: int = 500):
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    seq = seq.upper()
    if len(seq) < seq_len:
        seq = seq + "N" * (seq_len - len(seq))
    elif len(seq) > seq_len:
        seq = seq[:seq_len]
    arr = np.array([mapping.get(b, [0, 0, 0, 0]) for b in seq], dtype=np.float32)
    return arr.T


def revcomp_one_hot(x: np.ndarray):
    return x[::-1, ::-1].copy()


class SharedTrainModel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = nn.Conv1d(4, CONV_OUT, kernel_size=CONV_KERNEL)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        self.drop1 = nn.Dropout(0.2)

        self.td_dense = nn.Linear(CONV_OUT, 256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(0.2)

        pooled_len = (SEQ_LEN - CONV_KERNEL + 1) // 10
        self.fc1 = nn.Linear(512 * pooled_len, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.drop3 = nn.Dropout(0.4)

    def branch_forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.drop1(x)
        x = x.transpose(1, 2)
        x = torch.relu(self.td_dense(x))
        x, _ = self.lstm(x)
        x = self.drop2(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x

    def forward(self, x_fwd, x_rev):
        return (self.branch_forward(x_fwd) + self.branch_forward(x_rev)) / 2.0


def predict_probs(model, x_np: np.ndarray, device):
    x = torch.tensor(x_np[None, :, :], dtype=torch.float32, device=device)
    x_rc = torch.tensor(revcomp_one_hot(x_np)[None, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x, x_rc)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def mutate_onehot(x_np: np.ndarray, pos0: int, alt_base: str):
    base_to_row = {"A": 0, "C": 1, "G": 2, "T": 3}
    x_mut = x_np.copy()
    x_mut[:, pos0] = 0.0
    if alt_base in base_to_row:
        x_mut[base_to_row[alt_base], pos0] = 1.0
    return x_mut


def onehot_col_to_base(col: np.ndarray):
    idx = np.where(col == 1)[0]
    if len(idx) != 1:
        return "N"
    return BASES[int(idx[0])]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    seqs = read_fasta(FASTA)

    model = SharedTrainModel(n_classes=N_CLASSES).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    for window_id in WINDOW_IDS:
        if window_id not in seqs:
            print(f"[WARN] {window_id} not found, skipped.")
            continue

        seq = seqs[window_id]
        x_np = one_hot(seq, SEQ_LEN)
        ref_probs = predict_probs(model, x_np, device)

        rows = []
        for pos0 in range(SEQ_LEN):
            ref_base = onehot_col_to_base(x_np[:, pos0])
            if ref_base not in BASES:
                continue

            for alt_base in BASES:
                if alt_base == ref_base:
                    continue

                x_mut = mutate_onehot(x_np, pos0, alt_base)
                mut_probs = predict_probs(model, x_mut, device)

                rows.append({
                    "window_id": window_id,
                    "target_class": TARGET_CLASS,
                    "position_1based": pos0 + 1,
                    "ref_base": ref_base,
                    "alt_base": alt_base,
                    "ref_prob_0": float(ref_probs[0]),
                    "ref_prob_1": float(ref_probs[1]),
                    "ref_prob_2": float(ref_probs[2]),
                    "mut_prob_0": float(mut_probs[0]),
                    "mut_prob_1": float(mut_probs[1]),
                    "mut_prob_2": float(mut_probs[2]),
                    "delta_prob_0": float(mut_probs[0] - ref_probs[0]),
                    "delta_prob_1": float(mut_probs[1] - ref_probs[1]),
                    "delta_prob_2": float(mut_probs[2] - ref_probs[2]),
                    "delta_target_prob": float(mut_probs[TARGET_CLASS] - ref_probs[TARGET_CLASS]),
                })

        mut_df = pd.DataFrame(rows)
        mut_df.to_csv(OUTDIR / f"{window_id}.saturation_mutagenesis.tsv", sep="\t", index=False)

        min_delta = (
            mut_df.groupby("position_1based", as_index=False)["delta_target_prob"]
            .min()
            .sort_values("delta_target_prob")
        )
        min_delta.to_csv(OUTDIR / f"{window_id}.min_delta_summary.tsv", sep="\t", index=False)

        with open(OUTDIR / f"{window_id}.ref_probs.txt", "w") as f:
            f.write(f"prob_0\t{ref_probs[0]}\n")
            f.write(f"prob_1\t{ref_probs[1]}\n")
            f.write(f"prob_2\t{ref_probs[2]}\n")

        print(f"[INFO] Done: {window_id}")
        print("[INFO] Ref probs:", ref_probs)
        print(min_delta.head(10).to_string(index=False))

    print("[INFO] Saved to:", OUTDIR)


if __name__ == "__main__":
    main()
