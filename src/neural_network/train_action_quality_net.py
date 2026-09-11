"""
Trains the multi-task network: a shared GRU encoder over a window of
normalized keypoints, with two heads --

  1. classification head: which of the 6 exercises is being performed
  2. score head: a 0-100 "quality of execution" score

The score head is trained against the SAME data-driven scores already
computed by evaluate_dataset_fixed_targets_*.py (collected into
quality_labels.csv by generate_quality_labels.py). This means the network
is NOT learning any clinical notion of "correct form" -- there is no such
ground truth in MMFi. It's learning to reproduce, directly from raw
keypoints, the same population-relative heuristic scoring this project
already used, and picking up on "this subject's own range of motion
relative to the rest of the population" as the label naturally implies.
That's an intentional, accepted simplification for this project -- see
the chat discussion this script came out of.

Prerequisites (run in order):
  1. generate_quality_labels.py   -> quality_labels.csv
  2. build_windowed_dataset.py    -> action_quality_dataset.npz
  3. this script                  -> action_quality_net.pt

This file lives in <PROJECT_ROOT>/src/train_action_quality_net.py.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import PROJECT_ROOT

DATASET_NPZ = os.path.join(PROJECT_ROOT, 'action_quality_dataset.npz')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'action_quality_net.pt')

# --- Hyperparameters -----------------------------------------------------
# Deliberately small: the whole point is that this trains in minutes on a
# CPU, given how little data MMFi provides (a few thousand windows at
# most). A bigger/deeper network would just overfit faster on a dataset
# this size, not learn more.
HIDDEN_SIZE = 64
NUM_LAYERS = 1
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3

# Relative weight of the regression (score) loss vs. the classification
# loss in the combined loss the network is trained on. 0.5 means
# classification is treated as the "primary" task and scoring as
# secondary -- tune this if the score predictions look poor (increase it)
# or if classification accuracy drops noticeably once the score head is
# added (decrease it).
SCORE_LOSS_WEIGHT = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WindowDataset(Dataset):
    def __init__(self, X, y_class, y_score):
        self.X = torch.from_numpy(X).float()
        self.y_class = torch.from_numpy(y_class).long()
        # Scores are stored as 0-100 in the dataset; rescale to 0-1 here so
        # the regression loss lives on a similar numeric scale to the
        # classification loss instead of being ~100x larger and dominating
        # the combined loss.
        self.y_score = torch.from_numpy(y_score).float() / 100.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_score[idx]


class ActionQualityNet(nn.Module):
    """Shared GRU encoder + two heads: exercise classification and
    execution-quality score, both computed from the same window of
    normalized keypoints (see keypoint_normalize.py)."""

    def __init__(self, input_size, num_classes, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, time, input_size)
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) -- final layer's last hidden state
        class_logits = self.classifier(last_hidden)
        score = torch.sigmoid(self.scorer(last_hidden)).squeeze(-1)  # (batch,), in [0, 1]
        return class_logits, score


def run_epoch(model, loader, optimizer=None, class_weights=None):
    """One pass over `loader`. Trains if `optimizer` is given, otherwise
    only evaluates (no gradient updates, dropout/batchnorm in eval mode --
    though this model has neither, kept for good practice/future-proofing).

    `class_weights`, if given, is a (num_classes,) tensor passed to
    CrossEntropyLoss so under-represented classes contribute proportionally
    more to the loss -- a safety net against class imbalance on top of the
    mirroring augmentation in build_windowed_dataset.py, in case some
    classes still end up with fewer usable windows than others (e.g. more
    subjects skipped for invalid keypoints on one action than another)."""
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
        score_abs_error_sum += (score_pred - y_score).abs().sum().item() * 100.0  # back to 0-100 scale

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'score_mae': score_abs_error_sum / total,  # mean absolute error, in score points (0-100)
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

    # Inverse-frequency class weights: a class with half as many windows as
    # average gets ~2x weight in the classification loss. Computed from the
    # TRAIN split only (val is just for monitoring, shouldn't influence
    # what the model is optimized for).
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
    model = ActionQualityNet(input_size=input_size, num_classes=len(class_names)).to(DEVICE)
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
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'window_length': window_length,
                'input_size': input_size,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
            }, MODEL_PATH)
            print(f"           -> new best val_loss ({best_val_loss:.4f}), saved checkpoint")

    print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")
    print("Next step: write a live-inference script that loads this checkpoint, "
          "slides the same WINDOW_LENGTH window over normalized webcam keypoints, "
          "and reports (predicted exercise, quality score) -- happy to help with "
          "that once you've checked these numbers look reasonable.")


if __name__ == '__main__':
    main()