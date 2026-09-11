"""
Second method for comparison: a Temporal Convolutional Network (TCN / 1D-CNN)
instead of the GRU used in train_action_quality_net.py.

WHY THIS IS A FAIR COMPARISON, NOT JUST A DIFFERENT SCRIPT:
- Same input: the exact same action_quality_dataset.npz produced by
  build_windowed_dataset.py (34-dim normalized keypoints, WINDOW_LENGTH
  frames per window).
- Same task framing: shared trunk -> (classification head, score head),
  same loss (CrossEntropy + SCORE_LOSS_WEIGHT * MSE), same class weights,
  same train/val split (by subject, already baked into the .npz).
- Same training loop / metrics (accuracy, score MAE) as
  train_action_quality_net.py, so numbers printed by both scripts can be
  compared directly, epoch by epoch.
- Only difference: the encoder. GRU = recurrent, processes the window
  step-by-step and keeps a hidden state. TCN = convolutional, processes
  the whole window with stacked dilated 1D convolutions and a residual
  connection per block, then a global average pool over time. This is the
  standard "recurrent vs convolutional" comparison in sequence modeling
  literature.

Checkpoint stores an 'arch': 'tcn' field (and the channel/kernel config)
so a shared loader (see realtime_inference_action_quality.py) can
reconstruct either model from its checkpoint without hardcoding which
architecture was used.

Prerequisites (same as train_action_quality_net.py):
  1. generate_quality_labels.py   -> quality_labels.csv
  2. build_windowed_dataset.py    -> action_quality_dataset.npz
  3. this script                  -> action_quality_tcn.pt

This file lives in <PROJECT_ROOT>/src/train_action_quality_tcn.py,
alongside train_action_quality_net.py.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import PROJECT_ROOT

DATASET_NPZ = os.path.join(PROJECT_ROOT, 'action_quality_dataset.npz')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'action_quality_tcn.pt')

# --- Hyperparameters -----------------------------------------------------
# Kept deliberately small, same spirit as the GRU script: this has to train
# in minutes on a CPU, given how little data MMFi provides.
NUM_CHANNELS = [64, 64, 64]  # one entry per TCN block; dilation doubles each block (1, 2, 4, ...)
KERNEL_SIZE = 3
DROPOUT = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3

# Same relative weighting between the regression (score) loss and the
# classification loss as train_action_quality_net.py -- kept identical on
# purpose so any difference in results comes from the encoder, not from a
# different training objective.
SCORE_LOSS_WEIGHT = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WindowDataset(Dataset):
    """Identical to the one in train_action_quality_net.py."""
    def __init__(self, X, y_class, y_score):
        self.X = torch.from_numpy(X).float()
        self.y_class = torch.from_numpy(y_class).long()
        self.y_score = torch.from_numpy(y_score).float() / 100.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_score[idx]


class TemporalBlock(nn.Module):
    """One residual TCN block: two dilated 1D convolutions (same padding,
    so the sequence length is preserved) + BatchNorm + ReLU + Dropout,
    with a residual (skip) connection. A 1x1 conv on the skip path handles
    the case where the channel count changes between blocks."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # keeps time length constant
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                            if in_channels != out_channels else None)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + residual)


class ActionQualityTCN(nn.Module):
    """Shared TCN encoder + two heads: exercise classification and
    execution-quality score. Mirrors ActionQualityNet's interface
    (forward returns (class_logits, score)) so both models can be driven
    by the exact same run_epoch()/inference code."""

    def __init__(self, input_size, num_classes, num_channels=NUM_CHANNELS,
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        blocks = []
        in_ch = input_size
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i  # 1, 2, 4, ... -> grows the receptive field with depth
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_ch, num_classes)
        self.scorer = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x: (batch, time, input_size) -> Conv1d wants (batch, channels, time)
        x = x.transpose(1, 2)
        features = self.network(x)          # (batch, channels, time)
        pooled = features.mean(dim=2)       # global average pool over time
        class_logits = self.classifier(pooled)
        score = torch.sigmoid(self.scorer(pooled)).squeeze(-1)
        return class_logits, score


def run_epoch(model, loader, optimizer=None, class_weights=None):
    """Identical logic to train_action_quality_net.py's run_epoch, so the
    printed metrics are computed exactly the same way for both models."""
    is_training = optimizer is not None
    model.train(is_training)

    classification_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    regression_loss_fn = nn.MSELoss()

    total_loss = 0.0
    correct = 0
    total = 0
    score_abs_error_sum = 0.0

    for X, y_class, y_score in loader:
        X, y_class, y_score = X.to(DEVICE), y_class.to(DEVICE), y_score.to(DEVICE)

        with torch.set_grad_enabled(is_training):
            class_logits, score_pred = model(X)
            class_loss = classification_loss_fn(class_logits, y_class)
            score_loss = regression_loss_fn(score_pred, y_score)
            loss = class_loss + SCORE_LOSS_WEIGHT * score_loss

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        correct += (class_logits.argmax(dim=1) == y_class).sum().item()
        total += batch_size
        score_abs_error_sum += (score_pred - y_score).abs().sum().item() * 100.0

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'score_mae': score_abs_error_sum / total,
    }


def main():
    if not os.path.exists(DATASET_NPZ):
        print(f"ERROR: {DATASET_NPZ} not found. Run build_windowed_dataset.py first.")
        return

    data = np.load(DATASET_NPZ, allow_pickle=True)
    class_names = list(data['class_names'])
    window_length = int(data['window_length'])

    train_ds = WindowDataset(data['X_train'], data['y_class_train'], data['y_score_train'])
    val_ds = WindowDataset(data['X_val'], data['y_class_val'], data['y_score_val'])
    print(f"Device: {DEVICE}")
    print(f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}")
    print(f"Classes ({len(class_names)}): {class_names}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Same inverse-frequency class weighting as the GRU script, computed
    # from TRAIN only, so both models see the same effective loss surface.
    class_counts = np.bincount(data['y_class_train'], minlength=len(class_names)).astype(np.float32)
    class_weights = torch.tensor(
        class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1.0)),
        dtype=torch.float32
    ).to(DEVICE)
    print("Train windows per class:")
    for name, count, weight in zip(class_names, class_counts, class_weights.cpu().numpy()):
        print(f"  {name:22s} {int(count):5d} windows  (loss weight={weight:.2f})")
    print()

    input_size = train_ds.X.shape[-1]  # 34 = 17 keypoints * 2 coords
    model = ActionQualityTCN(input_size=input_size, num_classes=len(class_names)).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TCN, channels={NUM_CHANNELS}, kernel_size={KERNEL_SIZE}, "
          f"params={num_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, class_weights=class_weights)
        val_metrics = run_epoch(model, val_loader, optimizer=None, class_weights=class_weights)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']*100:5.1f}% "
              f"score_mae={train_metrics['score_mae']:4.1f}  |  "
              f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']*100:5.1f}% "
              f"score_mae={val_metrics['score_mae']:4.1f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'arch': 'tcn',
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'window_length': window_length,
                'input_size': input_size,
                'num_channels': NUM_CHANNELS,
                'kernel_size': KERNEL_SIZE,
                'dropout': DROPOUT,
            }, MODEL_PATH)
            print(f"           -> new best val_loss ({best_val_loss:.4f}), saved checkpoint")

    print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")
    print("Run compare_models.py to benchmark this checkpoint against "
          "action_quality_net.pt (the GRU) on the same validation split.")


if __name__ == '__main__':
    main()