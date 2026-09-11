"""
Compares the two trained action-quality models (GRU vs TCN) on the SAME
validation split from action_quality_dataset.npz, so the numbers are
directly comparable.

Run AFTER both:
  train_action_quality_net.py -> action_quality_net.pt   (GRU)
  train_action_quality_tcn.py -> action_quality_tcn.pt    (TCN)

Reports, for each model:
  - validation accuracy (classification head)
  - validation score MAE (regression head, 0-100 scale)
  - per-class accuracy (helps spot if one architecture struggles on a
    specific exercise, e.g. mixing up lunge_left / lunge_right)
  - number of parameters
  - average CPU inference time per single window (batch size 1), since
    that is what matters for the live webcam script running on a
    non-performant machine, not training time

This file lives in <PROJECT_ROOT>/src/compare_models.py, alongside
train_action_quality_net.py and train_action_quality_tcn.py.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, 'neural_network'))

from utils import PROJECT_ROOT, GRU_DIR, TCN_DIR
from train_action_quality_net import ActionQualityNet, WindowDataset as GRUWindowDataset
from train_action_quality_tcn import ActionQualityTCN, WindowDataset as TCNWindowDataset

# We pick the dataset from the GRU directory, but both architectures were trained on the same dataset
DATASET_NPZ = os.path.join(GRU_DIR, 'action_quality_dataset.npz')
GRU_CHECKPOINT = os.path.join(GRU_DIR, 'action_quality_net.pt')
TCN_CHECKPOINT = os.path.join(TCN_DIR, 'action_quality_tcn.pt')

# CPU on purpose: the comparison that matters for a non-performant machine
# is CPU inference speed, since that's what realtime_inference_action_quality.py
# actually runs on during a live session.
DEVICE = torch.device('cpu')

N_TIMING_RUNS = 200  # single-window forward passes used to estimate latency


def load_gru(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = ActionQualityNet(
        input_size=checkpoint['input_size'],
        num_classes=len(checkpoint['class_names']),
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def load_tcn(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = ActionQualityTCN(
        input_size=checkpoint['input_size'],
        num_classes=len(checkpoint['class_names']),
        num_channels=checkpoint['num_channels'],
        kernel_size=checkpoint['kernel_size'],
        dropout=checkpoint['dropout'],
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(model, loader, num_classes):
    correct = 0
    total = 0
    score_abs_error_sum = 0.0
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for X, y_class, y_score in loader:
        X, y_class, y_score = X.to(DEVICE), y_class.to(DEVICE), y_score.to(DEVICE)
        class_logits, score_pred = model(X)
        preds = class_logits.argmax(dim=1)

        correct += (preds == y_class).sum().item()
        total += X.size(0)
        score_abs_error_sum += (score_pred - y_score).abs().sum().item() * 100.0

        for c in range(num_classes):
            mask = (y_class == c)
            class_total[c] += mask.sum().item()
            class_correct[c] += (preds[mask] == c).sum().item()

    per_class_acc = np.divide(class_correct, class_total,
                               out=np.zeros_like(class_correct), where=class_total > 0)

    return {
        'accuracy': correct / total,
        'score_mae': score_abs_error_sum / total,
        'per_class_acc': per_class_acc,
    }


@torch.no_grad()
def measure_latency(model, sample_window, n_runs=N_TIMING_RUNS):
    """Average wall-clock time for a single-window forward pass on CPU --
    the realistic per-frame cost during live inference (see
    realtime_inference_action_quality.py, which runs one forward pass per
    completed window)."""
    x = sample_window.unsqueeze(0)  # (1, window_length, input_size)

    # Warm-up (first call(s) can be slower due to lazy initialization).
    for _ in range(5):
        model(x)

    start = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0  # milliseconds per window


def main():
    if not os.path.exists(DATASET_NPZ):
        print(f"ERROR: {DATASET_NPZ} not found. Run build_windowed_dataset.py first.")
        return
    missing = [p for p in (GRU_CHECKPOINT, TCN_CHECKPOINT) if not os.path.exists(p)]
    if missing:
        print("ERROR: missing checkpoint(s):")
        for p in missing:
            print(f"  {p}")
        print("Train both models first (train_action_quality_net.py and "
              "train_action_quality_tcn.py).")
        return

    data = np.load(DATASET_NPZ, allow_pickle=True)
    class_names = list(data['class_names'])
    num_classes = len(class_names)

    gru_model, gru_ckpt = load_gru(GRU_CHECKPOINT)
    tcn_model, tcn_ckpt = load_tcn(TCN_CHECKPOINT)

    # Each architecture has its own WindowDataset class (they're identical
    # in behavior, just imported from their respective training scripts) --
    # using each model's own dataset wrapper keeps this script decoupled
    # from having to know both scale scores the same way if that ever
    # changes.
    gru_val_ds = GRUWindowDataset(data['X_val'], data['y_class_val'], data['y_score_val'])
    tcn_val_ds = TCNWindowDataset(data['X_val'], data['y_class_val'], data['y_score_val'])
    gru_val_loader = DataLoader(gru_val_ds, batch_size=64, shuffle=False)
    tcn_val_loader = DataLoader(tcn_val_ds, batch_size=64, shuffle=False)

    print(f"Validation windows: {len(gru_val_ds)}\n")

    gru_metrics = evaluate(gru_model, gru_val_loader, num_classes)
    tcn_metrics = evaluate(tcn_model, tcn_val_loader, num_classes)

    gru_params = sum(p.numel() for p in gru_model.parameters())
    tcn_params = sum(p.numel() for p in tcn_model.parameters())

    sample_window = gru_val_ds.X[0]  # same underlying data for both, just used for timing
    gru_latency_ms = measure_latency(gru_model, sample_window)
    tcn_latency_ms = measure_latency(tcn_model, sample_window)

    print("=== Overall comparison (CPU) ===")
    print(f"{'Metric':<28}{'GRU':>15}{'TCN':>15}")
    print(f"{'Accuracy':<28}{gru_metrics['accuracy']*100:>14.1f}%{tcn_metrics['accuracy']*100:>14.1f}%")
    print(f"{'Score MAE (0-100)':<28}{gru_metrics['score_mae']:>15.2f}{tcn_metrics['score_mae']:>15.2f}")
    print(f"{'Parameters':<28}{gru_params:>15,}{tcn_params:>15,}")
    print(f"{'Inference latency (ms/window)':<28}{gru_latency_ms:>15.2f}{tcn_latency_ms:>15.2f}")

    print("\n=== Per-class accuracy ===")
    print(f"{'Class':<24}{'GRU':>12}{'TCN':>12}")
    for i, name in enumerate(class_names):
        print(f"{name:<24}{gru_metrics['per_class_acc'][i]*100:>11.1f}%{tcn_metrics['per_class_acc'][i]*100:>11.1f}%")

    print("\nNOTE: both models were trained on the identical windows/labels/"
          "split/loss, so any gap above reflects the architecture choice "
          "(recurrent vs convolutional), not a difference in data or "
          "training setup.")


if __name__ == '__main__':
    main()