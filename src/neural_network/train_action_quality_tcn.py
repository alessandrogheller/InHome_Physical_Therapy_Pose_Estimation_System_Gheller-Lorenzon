"""
Train a Temporal Convolutional Network to classify exercises and predict quality.

Updated version: runtime data augmentation, higher dropout, weight decay,
LR scheduling, early stopping and gradient clipping, plus a checkpoint
selection criterion based on classification accuracy first (the biggest
weakness observed in the baseline comparison) and validation loss as a
tie-breaker. Everything else (dataset format, checkpoint keys, model
interface) stays compatible with compare_models.py and
evaluate_on_new_subjects_with_3_methods.py.

--- UPDATE (score-quality fix) ------------------------------------------
Same two changes as train_action_quality_net.py, kept identical on purpose
so any remaining GRU-vs-TCN difference still comes from the encoder, not
from a different training objective/selection rule:

1. SCORE_LOSS_WEIGHT raised 0.5 -> 1.2, so the shared encoder keeps
   getting gradient signal for the score head even once classification
   has converged.
2. Checkpoint selection now targets val score_mae first, with val accuracy
   as a tie-breaker (previously: accuracy first, loss as tie-breaker).
   The TCN's score spread was already healthier than the GRU's on the
   3-subject eval (std=21.8% vs 9.4%), but this keeps both training
   scripts consistent and should still help push score_mae down further.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import TCN_DIR
from keypoint_normalize import augment_window

DATASET_NPZ = os.path.join(TCN_DIR, 'action_quality_dataset.npz')
MODEL_PATH = os.path.join(TCN_DIR, 'action_quality_tcn.pt')

# --- Hyperparameters -----------------------------------------------------
# Kept deliberately small, same spirit as before: this has to train
# in minutes on a CPU, given how little data MMFi provides.
NUM_CHANNELS = [64, 64, 64]  # one entry per TCN block; dilation doubles each block (1, 2, 4, ...)
KERNEL_SIZE = 3
DROPOUT = 0.2                 # raised from 0.1: more regularization given the small dataset
BATCH_SIZE = 64
MAX_EPOCHS = 150              # upper bound; early stopping will normally stop earlier
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4           # new: L2 regularization on the optimizer
GRAD_CLIP_NORM = 5.0          # new: clip gradients to stabilize training
EARLY_STOP_PATIENCE = 15      # new: stop after this many epochs without improvement
LR_SCHEDULER_PATIENCE = 6     # new: epochs to wait before halving the LR on plateau

# Raised from 0.5, same rationale as the GRU script: classification
# saturates early, so a higher weight keeps the regression head training
# instead of being drowned out by an already-solved classification term.
SCORE_LOSS_WEIGHT = 1.2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WindowDataset(Dataset):
    """Store input windows and their class and quality targets.

    When augment=True (train split only), a fresh random augmentation
    (small rotation, scale jitter, Gaussian noise -- see
    keypoint_normalize.augment_window) is applied to each window every
    time it's drawn, instead of once at dataset-build time. This means
    the network sees a different perturbed version of the same window on
    every epoch, which acts as a much stronger regularizer than a single
    static mirrored copy.
    """
    def __init__(self, X, y_class, y_score, augment=False):
        # Keep X as float32 numpy (not a tensor yet): augmentation happens
        # per-sample in __getitem__, so we don't want a shared pre-converted
        # tensor here.
        self.X = X.astype(np.float32)
        self.y_class = torch.from_numpy(y_class).long()
        # Scale quality targets to [0, 1], matching the model output range.
        self.y_score = torch.from_numpy(y_score).float() / 100.0
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        window = self.X[idx]
        if self.augment:
            window = augment_window(window)
        return torch.from_numpy(window).float(), self.y_class[idx], self.y_score[idx]


class TemporalBlock(nn.Module):
    """Residual TCN block with dilated convolutions and skip connection."""

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
        # Extra dropout right before the heads, on top of the per-block
        # dropout already applied inside TemporalBlock.
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_ch, num_classes)
        self.scorer = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x: (batch, time, input_size) -> Conv1d wants (batch, channels, time)
        x = x.transpose(1, 2)
        features = self.network(x)          # (batch, channels, time)
        # Average the learned features over time before applying both heads.
        pooled = features.mean(dim=2)       # global average pool over time
        pooled = self.head_dropout(pooled)
        class_logits = self.classifier(pooled)
        score = torch.sigmoid(self.scorer(pooled)).squeeze(-1)
        return class_logits, score


def run_epoch(model, loader, optimizer=None, class_weights=None):
    """Train or evaluate the model for one pass over the data."""
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
            # Use the same combined objective as the GRU model.
            class_loss = classification_loss_fn(class_logits, y_class)
            score_loss = regression_loss_fn(score_pred, y_score)
            loss = class_loss + SCORE_LOSS_WEIGHT * score_loss

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                # New: clip gradients to avoid occasional unstable updates,
                # especially now that augmentation adds more input variance.
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
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

    # Load the same pre-built windows used by the GRU training script.
    data = np.load(DATASET_NPZ, allow_pickle=True)
    class_names = list(data['class_names'])
    window_length = int(data['window_length'])

    # Augmentation is applied only to the train split; validation must stay
    # on the original, unperturbed windows so metrics are comparable across
    # epochs and against the baseline model.
    train_ds = WindowDataset(data['X_train'], data['y_class_train'], data['y_score_train'], augment=True)
    val_ds = WindowDataset(data['X_val'], data['y_class_val'], data['y_score_val'], augment=False)
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
          f"dropout={DROPOUT}, params={num_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # New: reduce the learning rate when validation loss stops improving,
    # instead of training at a fixed LR for the whole run.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE
    )

    # Checkpoint selection now targets score_mae first, with accuracy as a
    # tie-breaker -- see module docstring. val_loss is still tracked for
    # visibility/logging but no longer drives the selection.
    best_score_mae = float('inf')
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Train while tracking validation score_mae (primary) and accuracy
    # (tie-breaker) for checkpoint selection, with early stopping.
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, class_weights=class_weights)
        val_metrics = run_epoch(model, val_loader, optimizer=None, class_weights=class_weights)
        scheduler.step(val_metrics['loss'])

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  lr={current_lr:.2e}  "
              f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']*100:5.1f}% "
              f"score_mae={train_metrics['score_mae']:4.1f}  |  "
              f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']*100:5.1f}% "
              f"score_mae={val_metrics['score_mae']:4.1f}")

        # Selection criterion: score_mae first (the metric we're trying to
        # improve now that classification is already solid), accuracy as a
        # tie-breaker so it doesn't regress while chasing a lower MAE.
        improved = (val_metrics['score_mae'] < best_score_mae) or (
            val_metrics['score_mae'] == best_score_mae and val_metrics['accuracy'] > best_val_acc
        )

        if improved:
            best_score_mae = val_metrics['score_mae']
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            # Save only the best-performing model on the validation set.
            # Keep all original keys so downstream scripts (compare_models.py,
            # evaluate_on_new_subjects_with_3_methods.py) keep working unchanged.
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
            print(f"           -> new best (val_score_mae={best_score_mae:.2f}, "
                  f"val_acc={best_val_acc*100:.1f}%, val_loss={best_val_loss:.4f}), saved checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")


if __name__ == '__main__':
    main()