#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import random
import math
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
WINDOW_TABLE = ROOT / "data" / "al_shared_bg_peak_windows.tsv"
WINDOW_FASTA = ROOT / "input" / "sequences" / "al_shared_bg_peak_windows.fa"

# =========================
# Hyperparameters
# =========================
SEED = 42

SEQ_LEN = 500
BATCH_SIZE = 128
MAX_EPOCHS = 100
MIN_EPOCHS = 12
EARLY_STOP_PATIENCE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

CONV_OUT = 256
CONV_KERNEL = 12
N_CLASSES = 3
LABELS = ["random_intergenic", "shared_common", "Al_specific"]
AL_CLASS_ID = 2

FOCAL_GAMMA = 1.5
CLASS_WEIGHTS = [1.0, 1.2, 2.4]

BALANCE_RANDOM_INTERGENIC = True

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
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


def load_peak_ids(path: Path):
    df = pd.read_csv(path, sep="\t")
    col = "peak_id" if "peak_id" in df.columns else df.columns[0]
    return set(df[col].astype(str).tolist())


# =========================
# Dataset
# =========================
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
        y = int(row["label_id"])
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(x_rc, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            wid,
            row["peak_id"],
        )


# =========================
# Model
# =========================
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


# =========================
# Loss
# =========================
class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            at = self.alpha[targets]
            loss = -at * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# =========================
# Metrics
# =========================
def multiclass_auc_metrics(y_true, probs, labels):
    rows = []
    for class_id, class_name in enumerate(labels):
        y_bin = (y_true == class_id).astype(int)
        p = probs[:, class_id]
        try:
            auroc = roc_auc_score(y_bin, p)
        except Exception:
            auroc = np.nan
        try:
            aupr = average_precision_score(y_bin, p)
        except Exception:
            aupr = np.nan
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "auROC": auroc,
            "auPR": aupr,
        })
    out = pd.DataFrame(rows)
    macro_auroc = out["auROC"].mean(skipna=True)
    macro_aupr = out["auPR"].mean(skipna=True)
    return out, macro_auroc, macro_aupr


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_total = 0

    all_true = []
    all_pred = []
    all_probs = []
    all_window_ids = []
    all_peak_ids = []

    with torch.no_grad():
        for xb, xrc, yb, wids, pids in loader:
            xb = xb.to(device, non_blocking=True)
            xrc = xrc.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb, xrc)
            loss = criterion(logits, yb)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            bs = yb.size(0)
            total_loss += loss.item() * bs
            n_total += bs

            all_true.append(yb.cpu().numpy())
            all_pred.append(preds)
            all_probs.append(probs)
            all_window_ids.extend(list(wids))
            all_peak_ids.extend(list(pids))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    probs = np.concatenate(all_probs)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    avg_loss = total_loss / max(n_total, 1)

    pred_df = pd.DataFrame({
        "window_id": all_window_ids,
        "peak_id": all_peak_ids,
        "true_label": y_true,
        "pred_label": y_pred,
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "prob_2": probs[:, 2],
    })

    return avg_loss, macro_f1, y_true, y_pred, probs, pred_df


def peak_level_eval(window_pred_df: pd.DataFrame):
    grouped = window_pred_df.groupby("peak_id", as_index=False).agg({
        "true_label": "first",
        "prob_0": "max",
        "prob_1": "max",
        "prob_2": "max",
    })
    probs = grouped[["prob_0", "prob_1", "prob_2"]].values
    y_true = grouped["true_label"].values
    y_pred = probs.argmax(axis=1)

    peak_pred_df = grouped.copy()
    peak_pred_df["pred_label"] = y_pred

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return y_true, y_pred, probs, macro_f1, peak_pred_df


def save_reports(prefix: str, y_true, y_pred, probs, labels, outdir: Path):
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    auc_df, macro_auroc, macro_aupr = multiclass_auc_metrics(y_true, probs, labels)

    with open(outdir / f"{prefix}_report.txt", "w") as f:
        f.write(report)

    cm_df.to_csv(outdir / f"{prefix}_confusion.tsv", sep="\t")
    auc_df.to_csv(outdir / f"{prefix}_auc_metrics.tsv", sep="\t", index=False)

    with open(outdir / f"{prefix}_auc_summary.txt", "w") as f:
        f.write(f"macro-auROC\t{macro_auroc}\n")
        f.write(f"macro-auPR\t{macro_aupr}\n")
        al_row = auc_df[auc_df["class_id"] == AL_CLASS_ID].iloc[0]
        f.write(f"Al_specific_auROC\t{al_row['auROC']}\n")
        f.write(f"Al_specific_auPR\t{al_row['auPR']}\n")


# =========================
# Data preparation
# =========================
def prepare_window_splits(fold_dir: Path):
    windows = pd.read_csv(WINDOW_TABLE, sep="\t")
    seq_dict = read_fasta(WINDOW_FASTA)

    train_peaks = load_peak_ids(fold_dir / "train_peaks.tsv")
    val_peaks = load_peak_ids(fold_dir / "val_peaks.tsv")
    test_peaks = load_peak_ids(fold_dir / "test_peaks.tsv")

    windows = windows[windows["window_id"].isin(seq_dict.keys())].copy()

    train_df = windows[windows["peak_id"].astype(str).isin(train_peaks)].copy()
    val_df = windows[windows["peak_id"].astype(str).isin(val_peaks)].copy()
    test_df = windows[windows["peak_id"].astype(str).isin(test_peaks)].copy()

    return train_df, val_df, test_df, seq_dict


def downsample_random_intergenic(train_df: pd.DataFrame, seed: int = 42):
    if not BALANCE_RANDOM_INTERGENIC:
        return train_df.copy()

    df0 = train_df[train_df["label_id"] == 0].copy()
    df1 = train_df[train_df["label_id"] == 1].copy()
    df2 = train_df[train_df["label_id"] == 2].copy()

    target_n = min(len(df0), len(df1), len(df2))

    df0_ds = df0.sample(n=target_n, random_state=seed, replace=False)
    df1_ds = df1.sample(n=target_n, random_state=seed, replace=False)
    df2_ds = df2.sample(n=target_n, random_state=seed, replace=False)

    out = (
        pd.concat([df0_ds, df1_ds, df2_ds], axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return out


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_dir", required=True, help="Directory containing train_peaks.tsv/val_peaks.tsv/test_peaks.tsv")
    parser.add_argument("--outdir", required=True, help="Output directory for this fold")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate fold inputs without training")
    args = parser.parse_args()

    fold_dir = Path(args.fold_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    required_inputs = [
        fold_dir / "train_peaks.tsv",
        fold_dir / "val_peaks.tsv",
        fold_dir / "test_peaks.tsv",
        WINDOW_TABLE,
        WINDOW_FASTA,
    ]

    for path in required_inputs:
        if not path.is_file():
            parser.error(f"Missing input file: {path}")

    if outdir.exists():
        parser.error(
            f"Output directory already exists: {outdir}. "
            "Choose a new directory."
        )

    if args.check_inputs:
        for path in required_inputs:
            print(f"[PASS] {path}")
        print(f"[PASS] Unused output path: {outdir}")
        print("[PASS] Input check completed; no training started.")
        return

    outdir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    print("[INFO] Fold dir:", fold_dir)
    print("[INFO] Outdir:", outdir)

    train_df, val_df, test_df, seq_dict = prepare_window_splits(fold_dir)

    print("[INFO] Original train windows:", train_df.shape[0])
    train_df_bal = downsample_random_intergenic(train_df, seed=args.seed)
    print("[INFO] Balanced train windows:", train_df_bal.shape[0])

    print("[INFO] Train class counts (original):")
    print(train_df["label_id"].value_counts().sort_index().to_dict())
    print("[INFO] Train class counts (balanced):")
    print(train_df_bal["label_id"].value_counts().sort_index().to_dict())

    print("[INFO] Val windows:", val_df.shape[0])
    print("[INFO] Test windows:", test_df.shape[0])

    train_ds = WindowDataset(train_df_bal, seq_dict)
    val_ds = WindowDataset(val_df, seq_dict)
    test_ds = WindowDataset(test_df, seq_dict)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = MediumCNNBiLSTMJasparMulti(n_classes=N_CLASSES).to(device)
    alpha = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    criterion = MultiClassFocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_metric = -math.inf
    best_epoch = -1
    best_state = None
    best_val_loss = math.inf
    patience_count = 0

    history_rows = []

    with open(outdir / "run_config.json", "w") as f:
        json.dump({
            "SEED": args.seed,
            "fold_dir": str(fold_dir),
            "SEQ_LEN": SEQ_LEN,
            "BATCH_SIZE": BATCH_SIZE,
            "MAX_EPOCHS": MAX_EPOCHS,
            "MIN_EPOCHS": MIN_EPOCHS,
            "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "NUM_WORKERS": NUM_WORKERS,
            "CONV_OUT": CONV_OUT,
            "CONV_KERNEL": CONV_KERNEL,
            "N_CLASSES": N_CLASSES,
            "LABELS": LABELS,
            "AL_CLASS_ID": AL_CLASS_ID,
            "FOCAL_GAMMA": FOCAL_GAMMA,
            "CLASS_WEIGHTS": CLASS_WEIGHTS,
            "BALANCE_RANDOM_INTERGENIC": BALANCE_RANDOM_INTERGENIC,
        }, f, indent=2)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_true = []
        train_pred = []

        for xb, xrc, yb, _, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            xrc = xrc.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb, xrc)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            bs = yb.size(0)
            train_loss_sum += loss.item() * bs
            train_n += bs
            train_true.append(yb.detach().cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss = train_loss_sum / max(train_n, 1)
        train_f1 = f1_score(train_true, train_pred, average="macro")

        val_loss, val_f1, y_true_v, y_pred_v, probs_v, _ = evaluate_model(model, val_loader, criterion, device)
        auc_df_v, macro_auroc_v, macro_aupr_v = multiclass_auc_metrics(y_true_v, probs_v, LABELS)
        val_al_aupr = float(auc_df_v.loc[auc_df_v["class_id"] == AL_CLASS_ID, "auPR"].iloc[0])
        val_al_auroc = float(auc_df_v.loc[auc_df_v["class_id"] == AL_CLASS_ID, "auROC"].iloc[0])

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_macro_auROC": macro_auroc_v,
            "val_macro_auPR": macro_aupr_v,
            "val_Al_specific_auROC": val_al_auroc,
            "val_Al_specific_auPR": val_al_aupr,
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} f1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} f1 {val_f1:.4f} | "
            f"val Al_auPR {val_al_aupr:.4f}"
        )

        if val_al_aupr > best_metric:
            best_metric = val_al_aupr
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch >= MIN_EPOCHS and patience_count >= EARLY_STOP_PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(outdir / "training_history.tsv", sep="\t", index=False)

    if best_state is None:
        raise RuntimeError("No best model was saved.")

    model.load_state_dict(best_state)
    torch.save(best_state, outdir / "best_model_by_val_Al_specific_auPR.pt")

    print(f"[INFO] Best epoch by val_Al_specific_auPR: {best_epoch}")
    print(f"[INFO] Best val_Al_specific_auPR: {best_metric:.6f}")
    print(f"[INFO] Corresponding val_loss: {best_val_loss:.6f}")

    train_df[["peak_id"]].drop_duplicates().to_csv(outdir / "train_peaks.tsv", sep="\t", index=False)
    val_df[["peak_id"]].drop_duplicates().to_csv(outdir / "val_peaks.tsv", sep="\t", index=False)
    test_df[["peak_id"]].drop_duplicates().to_csv(outdir / "test_peaks.tsv", sep="\t", index=False)

    test_loss, test_f1, y_true_w, y_pred_w, probs_w, test_pred_df = evaluate_model(
        model, test_loader, criterion, device
    )
    test_pred_df.to_csv(outdir / "test_window_predictions.tsv", sep="\t", index=False)
    save_reports("test_window", y_true_w, y_pred_w, probs_w, LABELS, outdir)

    y_true_p, y_pred_p, probs_p, peak_f1, peak_pred_df = peak_level_eval(test_pred_df)
    peak_pred_df.to_csv(outdir / "test_peak_predictions_maxpool.tsv", sep="\t", index=False)
    save_reports("test_peak", y_true_p, y_pred_p, probs_p, LABELS, outdir)

    with open(outdir / "test_window_report.txt", "w") as f:
        f.write(f"Best epoch by val_Al_specific_auPR: {best_epoch}\n")
        f.write(f"Best val_Al_specific_auPR: {best_metric:.6f}\n")
        f.write(f"Corresponding val_loss: {best_val_loss:.6f}\n")
        f.write(f"Window-level macro-F1: {test_f1:.6f}\n\n")
        f.write(classification_report(
            y_true_w, y_pred_w,
            target_names=LABELS,
            digits=4,
            zero_division=0
        ))

    with open(outdir / "test_peak_report_maxpool.txt", "w") as f:
        f.write(f"Best epoch by val_Al_specific_auPR: {best_epoch}\n")
        f.write(f"Best val_Al_specific_auPR: {best_metric:.6f}\n")
        f.write(f"Corresponding val_loss: {best_val_loss:.6f}\n")
        f.write(f"Peak-level macro-F1 (max pooling): {peak_f1:.6f}\n\n")
        f.write(classification_report(
            y_true_p, y_pred_p,
            target_names=LABELS,
            digits=4,
            zero_division=0
        ))

    auc_w, macro_auroc_w, macro_aupr_w = multiclass_auc_metrics(y_true_w, probs_w, LABELS)
    auc_p, macro_auroc_p, macro_aupr_p = multiclass_auc_metrics(y_true_p, probs_p, LABELS)
    al_w = auc_w[auc_w["class_id"] == AL_CLASS_ID].iloc[0]
    al_p = auc_p[auc_p["class_id"] == AL_CLASS_ID].iloc[0]

    print(f"[INFO] Window-level macro-auROC: {macro_auroc_w}")
    print(f"[INFO] Window-level macro-auPR: {macro_aupr_w}")
    print(f"[INFO] Window-level Al_specific auROC: {al_w['auROC']}")
    print(f"[INFO] Window-level Al_specific auPR: {al_w['auPR']}")
    print(f"[INFO] Peak-level macro-auROC: {macro_auroc_p}")
    print(f"[INFO] Peak-level macro-auPR: {macro_aupr_p}")
    print(f"[INFO] Peak-level Al_specific auROC: {al_p['auROC']}")
    print(f"[INFO] Peak-level Al_specific auPR: {al_p['auPR']}")
    print(f"[INFO] Saved to: {outdir}")


if __name__ == "__main__":
    main()
