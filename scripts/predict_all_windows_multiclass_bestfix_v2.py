#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
WINDOW_TABLE = ROOT / "data" / "al_shared_bg_peak_windows.tsv"
WINDOW_FASTA = ROOT / "input" / "sequences" / "al_shared_bg_peak_windows.fa"
MODEL_PATH = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42" / "best_model_by_val_Al_specific_auPR.pt"
OUTDIR = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 500
BATCH_SIZE = 128
NUM_WORKERS = 4
CONV_OUT = 256
CONV_KERNEL = 12
N_CLASSES = 3


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


class WindowDataset(Dataset):
    def __init__(self, windows_df: pd.DataFrame, seq_dict: dict):
        self.df = windows_df.reset_index(drop=True)
        self.seq_dict = seq_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wid = row["window_id"]
        seq = self.seq_dict[wid]
        x = one_hot(seq, SEQ_LEN)
        x_rc = revcomp_one_hot(x)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(x_rc, dtype=torch.float32),
            wid,
            row["peak_id"],
            row["chrom"],
            row["start"],
            row["end"],
            row["label"],
            row["label_id"],
        )


class MediumCNNBiLSTMJasparMulti(nn.Module):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    windows = pd.read_csv(WINDOW_TABLE, sep="\t")
    seq_dict = read_fasta(WINDOW_FASTA)
    windows = windows[windows["window_id"].isin(seq_dict.keys())].copy()

    ds = WindowDataset(windows, seq_dict)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = MediumCNNBiLSTMJasparMulti(n_classes=N_CLASSES).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rows = []
    with torch.no_grad():
        for xb, xrc, wids, pids, chroms, starts, ends, labels, label_ids in loader:
            xb = xb.to(device, non_blocking=True)
            xrc = xrc.to(device, non_blocking=True)
            logits = model(xb, xrc)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            for i in range(len(wids)):
                rows.append({
                    "window_id": wids[i],
                    "peak_id": pids[i],
                    "chrom": chroms[i],
                    "start": int(starts[i]),
                    "end": int(ends[i]),
                    "label": labels[i],
                    "label_id": int(label_ids[i]),
                    "pred_label": int(preds[i]),
                    "prob_0": float(probs[i, 0]),
                    "prob_1": float(probs[i, 1]),
                    "prob_2": float(probs[i, 2]),
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "all_window_predictions.tsv", sep="\t", index=False)
    print("[INFO] Saved:", OUTDIR / "all_window_predictions.tsv")
    print("[INFO] Total windows:", len(out))
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
