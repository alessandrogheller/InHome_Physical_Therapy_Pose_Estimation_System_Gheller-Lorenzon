"""
Train a multi-task GRU to classify exercises and predict execution quality.
Versione aggiornata: augmentation a runtime, dropout, weight decay,
LR scheduler, early stopping, gradient clipping, selezione checkpoint
basata anche sull'accuratezza di classificazione.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import GRU_DIR
from keypoint_normalize import augment_window

DATASET_NPZ = os.path.join(GRU_DIR, 'action_quality_dataset.npz')
MODEL_PATH = os.path.join(GRU_DIR, 'action_quality_net.pt')

# --- Hyperparameters -----------------------------------------------------
HIDDEN_SIZE = 64
NUM_LAYERS = 2          # prima: 1. Con dropout inter-layer aiuta la generalizzazione.
DROPOUT = 0.3           # nuovo: dropout tra i layer GRU + prima delle teste.
BATCH_SIZE = 64
MAX_EPOCHS = 150        # tetto massimo; l'early stopping fermerà prima.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4     # nuovo: L2 regularization.
GRAD_CLIP_NORM = 5.0    # nuovo.
EARLY_STOP_PATIENCE = 15  # nuovo: epoche senza miglioramento prima di fermarsi.
LR_SCHEDULER_PATIENCE = 6

SCORE_LOSS_WEIGHT = 0.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WindowDataset(Dataset):
    def __init__(self, X, y_class, y_score, augment=False):
        self.X = X.astype(np.float32)          # tenuto come numpy: l'augmentation agisce qui
        self.y_class = torch.from_numpy(y_class).long()
        self.y_score = torch.from_numpy(y_score).float() / 100.0
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        window = self.X[idx]
        if self.augment:
            window = augment_window(window)
        return torch.from_numpy(window).float(), self.y_class[idx], self.y_score[idx]


class ActionQualityNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.scorer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        last_hidden = self.dropout(h_n[-1])
        class_logits = self.classifier(last_hidden)
        score = torch.sigmoid(self.scorer(last_hidden)).squeeze(-1)
        return class_logits, score


def run_epoch(model, loader, optimizer=None, class_weights=None):
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

    data = np.load(DATASET_NPZ, allow_pickle=True)
    class_names = list(data['class_names'])
    window_length = int(data['window_length'])

    # augment=True solo sul train set
    train_ds = WindowDataset(data['X_train'], data['y_class_train'], data['y_score_train'], augment=True)
    val_ds = WindowDataset(data['X_val'], data['y_class_val'], data['y_score_val'], augment=False)
    print(f"Device: {DEVICE}")
    print(f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}")
    print(f"Classes ({len(class_names)}): {class_names}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_counts = np.bincount(data['y_class_train'], minlength=len(class_names)).astype(np.float32)
    class_weights = torch.tensor(
        class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1.0)),
        dtype=torch.float32
    ).to(DEVICE)
    print("Train windows per class:")
    for name, count, weight in zip(class_names, class_counts, class_weights.cpu().numpy()):
        print(f"  {name:22s} {int(count):5d} windows  (loss weight={weight:.2f})")
    print()

    input_size = train_ds.X.shape[-1]
    model = ActionQualityNet(input_size=input_size, num_classes=len(class_names)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, class_weights=class_weights)
        val_metrics = run_epoch(model, val_loader, optimizer=None, class_weights=class_weights)
        scheduler.step(val_metrics['loss'])

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']*100:5.1f}% "
              f"score_mae={train_metrics['score_mae']:4.1f}  |  "
              f"val: loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']*100:5.1f}% "
              f"score_mae={val_metrics['score_mae']:4.1f}")

        # Criterio di selezione: prima l'accuratezza di classificazione
        # (il problema piu' grave osservato), poi la loss come tie-breaker.
        improved = (val_metrics['accuracy'] > best_val_acc) or (
            val_metrics['accuracy'] == best_val_acc and val_metrics['loss'] < best_val_loss
        )

        if improved:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'window_length': window_length,
                'input_size': input_size,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
            }, MODEL_PATH)
            print(f"           -> new best (val_acc={best_val_acc*100:.1f}%, "
                  f"val_loss={best_val_loss:.4f}), saved checkpoint")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping: nessun miglioramento da {EARLY_STOP_PATIENCE} epoche.")
                break

    print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")


if __name__ == '__main__':
    main()