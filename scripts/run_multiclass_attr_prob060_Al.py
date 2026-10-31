#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import DeepLiftShap

ROOT = Path(__file__).resolve().parents[1]
FASTA = ROOT / "input" / "sequences" / "al_shared_bg_peak_windows.fa"
MODEL_PATH = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42" / "best_model_by_val_Al_specific_auPR.pt"

TOP_TABLE = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42" / "all_truepred_Al_top1_per_peak_prob060.tsv"

OUTDIR = ROOT / "results" / "al_shared_bg_multiclass_jaspar_bestfix_k12_g15_w124_balanced" / "seed_42" / "prob060_Al_attr"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 500
CONV_OUT = 256
CONV_KERNEL = 12
N_CLASSES = 3
TARGET_CLASS = 2

N_BASELINES = 20
SEED = 42


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def make_shuffle_baselines(x_np: np.ndarray, n_baselines: int = 20):
    baselines = []
    L = x_np.shape[1]
    for _ in range(n_baselines):
        perm = np.random.permutation(L)
        baselines.append(x_np[:, perm].copy())
    return np.stack(baselines, axis=0)


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


class NonSharedBranch(nn.Module):
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

    def forward(self, x):
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


class AttributionModel(nn.Module):
    def __init__(self, fwd_branch, rev_branch, target_class=2):
        super().__init__()
        self.fwd_branch = fwd_branch
        self.rev_branch = rev_branch
        self.target_class = target_class

    def forward(self, x_fwd, x_rev):
        logits = (self.fwd_branch(x_fwd) + self.rev_branch(x_rev)) / 2.0
        probs = torch.softmax(logits, dim=1)
        return probs[:, self.target_class]


def copy_shared_to_nonshared(shared_model: SharedTrainModel):
    fwd = NonSharedBranch(n_classes=N_CLASSES)
    rev = NonSharedBranch(n_classes=N_CLASSES)

    for branch in [fwd, rev]:
        branch.conv.load_state_dict(copy.deepcopy(shared_model.conv.state_dict()))
        branch.td_dense.load_state_dict(copy.deepcopy(shared_model.td_dense.state_dict()))
        branch.lstm.load_state_dict(copy.deepcopy(shared_model.lstm.state_dict()))
        branch.fc1.load_state_dict(copy.deepcopy(shared_model.fc1.state_dict()))
        branch.fc2.load_state_dict(copy.deepcopy(shared_model.fc2.state_dict()))

    return fwd, rev


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print("[INFO] Top table:", TOP_TABLE)
    print("[INFO] Output dir:", OUTDIR)

    top_df = pd.read_csv(TOP_TABLE, sep="\t")
    window_ids = top_df["window_id"].astype(str).tolist()

    seqs = read_fasta(FASTA)

    shared_model = SharedTrainModel(n_classes=N_CLASSES).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    shared_model.load_state_dict(state)
    shared_model.eval()

    fwd_branch, rev_branch = copy_shared_to_nonshared(shared_model)
    fwd_branch = fwd_branch.to(device).eval()
    rev_branch = rev_branch.to(device).eval()

    attr_model = AttributionModel(
        fwd_branch,
        rev_branch,
        target_class=TARGET_CLASS
    ).to(device).eval()

    explainer = DeepLiftShap(attr_model)

    all_actual_contrib = []
    summary_rows = []

    for idx, window_id in enumerate(window_ids, start=1):
        if window_id not in seqs:
            print(f"[WARN] {window_id} not found in FASTA, skipped.")
            continue

        seq = seqs[window_id]
        x_np = one_hot(seq, SEQ_LEN)
        x_rc_np = revcomp_one_hot(x_np)

        x = torch.tensor(x_np[None, :, :], dtype=torch.float32, device=device)
        x_rc = torch.tensor(x_rc_np[None, :, :], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = shared_model(x, x_rc)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        baselines_fwd = make_shuffle_baselines(x_np, n_baselines=N_BASELINES)
        baselines_rev = np.stack([revcomp_one_hot(b) for b in baselines_fwd], axis=0)

        b_fwd = torch.tensor(baselines_fwd, dtype=torch.float32, device=device)
        b_rev = torch.tensor(baselines_rev, dtype=torch.float32, device=device)

        attrs = explainer.attribute(
            inputs=(x, x_rc),
            baselines=(b_fwd, b_rev),
        )

        attr_fwd = attrs[0].detach().cpu().numpy()[0]  # 4 x 500
        actual_contrib = attr_fwd * x_np               # 4 x 500

        contrib_1d = actual_contrib.sum(axis=0)

        max_abs_pos = int(np.argmax(np.abs(contrib_1d))) + 1
        max_abs_importance = float(np.max(np.abs(contrib_1d)))
        mean_abs_importance = float(np.mean(np.abs(contrib_1d)))
        sum_positive = float(np.sum(contrib_1d[contrib_1d > 0])) if np.any(contrib_1d > 0) else 0.0
        sum_negative = float(np.sum(contrib_1d[contrib_1d < 0])) if np.any(contrib_1d < 0) else 0.0

        center = max_abs_pos - 1
        left = max(0, center - 10)
        right = min(SEQ_LEN, center + 11)
        local_seq = seq[left:right]

        np.save(OUTDIR / f"{window_id}.attr.npy", attr_fwd)
        np.save(OUTDIR / f"{window_id}.actual_contrib.npy", actual_contrib)

        row_meta = top_df[top_df["window_id"] == window_id].iloc[0].to_dict()
        summary_rows.append({
            **row_meta,
            "target_class": TARGET_CLASS,
            "model_prob_0": float(probs[0]),
            "model_prob_1": float(probs[1]),
            "model_prob_2": float(probs[2]),
            "max_abs_position": max_abs_pos,
            "max_abs_importance": max_abs_importance,
            "mean_abs_importance": mean_abs_importance,
            "sum_positive_importance": sum_positive,
            "sum_negative_importance": sum_negative,
            "local_seq_21bp": local_seq,
        })

        all_actual_contrib.append(actual_contrib)

        if idx % 10 == 0 or idx == 1:
            print(f"[INFO] {idx}/{len(window_ids)} done: {window_id}, prob_2={probs[2]:.4f}")

    if len(all_actual_contrib) == 0:
        raise RuntimeError("No attribution results generated.")

    all_actual_contrib = np.stack(all_actual_contrib, axis=0)  # N x 4 x 500
    mean_actual_contrib = all_actual_contrib.mean(axis=0)      # 4 x 500

    np.save(OUTDIR / "prob060_actual_contrib_all.npy", all_actual_contrib)
    np.save(OUTDIR / "prob060_actual_contrib_mean.npy", mean_actual_contrib)

    bases = ["A", "C", "G", "T"]
    mean_df = pd.DataFrame({
        "position_1based": np.arange(1, SEQ_LEN + 1),
        "A": mean_actual_contrib[0, :],
        "C": mean_actual_contrib[1, :],
        "G": mean_actual_contrib[2, :],
        "T": mean_actual_contrib[3, :],
        "sum_actual_contrib": mean_actual_contrib.sum(axis=0),
        "mean_abs_actual_contrib": np.mean(np.abs(all_actual_contrib), axis=(0, 1)),
    })
    mean_df.to_csv(OUTDIR / "prob060_actual_contrib_mean.tsv", sep="\t", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTDIR / "prob060_attr_summary.tsv", sep="\t", index=False)

    print("[INFO] Saved:", OUTDIR / "prob060_actual_contrib_all.npy")
    print("[INFO] Saved:", OUTDIR / "prob060_actual_contrib_mean.npy")
    print("[INFO] Saved:", OUTDIR / "prob060_actual_contrib_mean.tsv")
    print("[INFO] Saved:", OUTDIR / "prob060_attr_summary.tsv")
    print("[INFO] Finished.")


if __name__ == "__main__":
    main()
