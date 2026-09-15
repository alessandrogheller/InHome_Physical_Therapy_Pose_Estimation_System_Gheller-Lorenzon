"""
src/evaluate_dataset/plot_three_methods_comparison.py

Reads the CSV produced by evaluate_on_new_subjects_with_3_methods.py
(three_methods_comparison.csv) and produces report-ready figures comparing
the rule-based, GRU, and TCN quality scores across subjects/exercises.

Matches the CSV produced by evaluate_three_methods.py:
    subject, exercise, n_frames,
    rule_based_score,
    gru_score, gru_pred_class, gru_pred_conf, gru_class_correct,
    tcn_score, tcn_pred_class, tcn_pred_conf, tcn_class_correct

If you rename columns in evaluate_three_methods.py, just edit the COLUMN
MAPPING section right below the imports -- everything else works off
those names, nothing is hardcoded elsewhere.

Usage:
    python plot_three_methods_comparison.py path/to/three_methods_comparison.csv

Output: a set of .png files saved next to the input CSV (or in --outdir).
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- COLUMN MAPPING: matches evaluate_three_methods.py's CSV output --------
COL_SUBJECT = 'subject'
COL_EXERCISE = 'exercise'
COL_RULE_SCORE = 'rule_based_score'
COL_GRU_SCORE = 'gru_score'
COL_GRU_PRED = 'gru_pred_class'
COL_GRU_CORRECT = 'gru_class_correct'
COL_TCN_SCORE = 'tcn_score'
COL_TCN_PRED = 'tcn_pred_class'
COL_TCN_CORRECT = 'tcn_class_correct'

METHOD_COLORS = {'Rule-based': '#4C72B0', 'GRU': '#DD8452', 'TCN': '#55A868'}


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    required = [COL_SUBJECT, COL_EXERCISE, COL_RULE_SCORE, COL_GRU_SCORE, COL_TCN_SCORE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected column(s) {missing} in {csv_path}. "
            f"Found columns: {list(df.columns)}. "
            "Edit the COLUMN MAPPING section at the top of this script to match your CSV."
        )
    df['label'] = df[COL_SUBJECT].astype(str) + '/' + df[COL_EXERCISE].astype(str)

    # A method can legitimately fail on a given video (e.g. sequence too
    # short to form one window) -- evaluate_three_methods.py leaves that
    # cell empty, which pandas reads as NaN. Report it instead of silently
    # letting it break a plot.
    for col, name in [(COL_RULE_SCORE, 'rule-based'), (COL_GRU_SCORE, 'GRU'), (COL_TCN_SCORE, 'TCN')]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"Note: {n_missing} row(s) have no {name} score (method failed on that video) -- "
                  f"excluded from plots that need it.")
    return df


# --- 1. Grouped bar chart: mean score per exercise, averaged across subjects
def plot_grouped_bars(df, outdir):
    exercises = sorted(df[COL_EXERCISE].unique())

    means = {m: [] for m in ('rule', 'gru', 'tcn')}
    stds = {m: [] for m in ('rule', 'gru', 'tcn')}
    counts = []

    for ex in exercises:
        sub = df[df[COL_EXERCISE] == ex]
        for key, col in (('rule', COL_RULE_SCORE), ('gru', COL_GRU_SCORE), ('tcn', COL_TCN_SCORE)):
            vals = sub[col].dropna()
            means[key].append(vals.mean() if len(vals) > 0 else np.nan)
            stds[key].append(vals.std() if len(vals) > 1 else 0.0)
        counts.append(sub[COL_RULE_SCORE].notna().sum())  # subjects contributing, roughly

    x = np.arange(len(exercises))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(exercises) * 1.6), 6))
    ax.bar(x - width, means['rule'], width, yerr=stds['rule'], capsize=4,
           label='Rule-based', color=METHOD_COLORS['Rule-based'])
    ax.bar(x, means['gru'], width, yerr=stds['gru'], capsize=4,
           label='GRU', color=METHOD_COLORS['GRU'])
    ax.bar(x + width, means['tcn'], width, yerr=stds['tcn'], capsize=4,
           label='TCN', color=METHOD_COLORS['TCN'])

    xtick_labels = [f"{ex}\n(n={n})" for ex, n in zip(exercises, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Quality score (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Mean quality score per exercise, averaged across subjects\n(error bars = std across subjects)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'scores_grouped_bars_mean_per_exercise.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# --- 2. Heatmaps: subject x exercise, one panel per method -----------------
def plot_heatmaps(df, outdir):
    subjects = sorted(df[COL_SUBJECT].unique())
    exercises = sorted(df[COL_EXERCISE].unique())

    methods = [('Rule-based', COL_RULE_SCORE), ('GRU', COL_GRU_SCORE), ('TCN', COL_TCN_SCORE)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4 + 0.3 * len(subjects)))

    for ax, (name, col) in zip(axes, methods):
        matrix = np.full((len(subjects), len(exercises)), np.nan)
        for _, row in df.iterrows():
            i = subjects.index(row[COL_SUBJECT])
            j = exercises.index(row[COL_EXERCISE])
            matrix[i, j] = row[col]

        im = ax.imshow(matrix, vmin=0, vmax=100, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(exercises)))
        ax.set_xticklabels(exercises, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(subjects)))
        ax.set_yticklabels(subjects, fontsize=8)
        ax.set_title(name)
        for i in range(len(subjects)):
            for j in range(len(exercises)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.0f}", ha='center', va='center', fontsize=7)

    fig.colorbar(im, ax=axes, shrink=0.8, label='Score (%)')
    fig.suptitle('Score heatmap by subject and exercise, per method')
    path = os.path.join(outdir, 'scores_heatmaps.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# --- 3. Pairwise agreement scatter plots -----------------------------------
def plot_agreement(df, outdir):
    pairs = [
        ('Rule-based', COL_RULE_SCORE, 'GRU', COL_GRU_SCORE),
        ('Rule-based', COL_RULE_SCORE, 'TCN', COL_TCN_SCORE),
        ('GRU', COL_GRU_SCORE, 'TCN', COL_TCN_SCORE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name_a, col_a, name_b, col_b) in zip(axes, pairs):
        paired = df.dropna(subset=[col_a, col_b])
        a, b = paired[col_a].values, paired[col_b].values
        r = np.corrcoef(a, b)[0, 1] if len(a) > 1 else float('nan')
        mad = np.mean(np.abs(a - b)) if len(a) > 0 else float('nan')

        ax.scatter(a, b, alpha=0.8, color='#4C72B0')
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='y = x (perfect agreement)')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel(f'{name_a} score (%)')
        ax.set_ylabel(f'{name_b} score (%)')
        r_str = f"{r:.2f}" if not np.isnan(r) else "n/a"
        mad_str = f"{mad:.1f}" if not np.isnan(mad) else "n/a"
        ax.set_title(f'{name_a} vs {name_b}  (n={len(a)})\nr={r_str}, mean|diff|={mad_str} pts')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, 'pairwise_agreement.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# --- 4. Confusion matrices for GRU / TCN exercise classification -----------
def plot_confusion_matrices(df, outdir):
    if COL_GRU_PRED not in df.columns and COL_TCN_PRED not in df.columns:
        print("No prediction columns found, skipping confusion matrices.")
        return

    exercises = sorted(df[COL_EXERCISE].unique())
    idx = {e: i for i, e in enumerate(exercises)}

    preds_available = [(name, col, correct_col) for name, col, correct_col in
                        [('GRU', COL_GRU_PRED, COL_GRU_CORRECT), ('TCN', COL_TCN_PRED, COL_TCN_CORRECT)]
                        if col in df.columns]
    fig, axes = plt.subplots(1, len(preds_available), figsize=(6 * len(preds_available), 5))
    if len(preds_available) == 1:
        axes = [axes]

    for ax, (name, col, correct_col) in zip(axes, preds_available):
        cm = np.zeros((len(exercises), len(exercises)), dtype=int)
        valid_rows = df.dropna(subset=[col])
        for _, row in valid_rows.iterrows():
            true_i = idx[row[COL_EXERCISE]]
            pred_j = idx.get(row[col], None)
            if pred_j is not None:
                cm[true_i, pred_j] += 1

        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(exercises)))
        ax.set_xticklabels(exercises, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(exercises)))
        ax.set_yticklabels(exercises, fontsize=8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True (folder name)')
        # Prefer the pre-computed *_class_correct flags (written by
        # evaluate_three_methods.py) over re-deriving accuracy from the
        # matrix, so this number always matches what the main script printed.
        if correct_col in df.columns and df[correct_col].notna().any():
            acc = df[correct_col].dropna().astype(bool).mean()
        else:
            acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else float('nan')
        ax.set_title(f'{name} confusion matrix (accuracy={acc*100:.1f}%)')
        for i in range(len(exercises)):
            for j in range(len(exercises)):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                             color='white' if cm[i, j] > cm.max() / 2 else 'black')

    fig.tight_layout()
    path = os.path.join(outdir, 'confusion_matrices.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# --- 5. Boxplot of score distributions per method ---------------------------
def plot_boxplot(df, outdir):
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [df[COL_RULE_SCORE].dropna(), df[COL_GRU_SCORE].dropna(), df[COL_TCN_SCORE].dropna()]
    labels = ['Rule-based', 'GRU', 'TCN']
    try:
        # Matplotlib >= 3.9 renamed 'labels' to 'tick_labels'.
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    except TypeError:
        # Older matplotlib doesn't know 'tick_labels' yet -- fall back.
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], METHOD_COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Quality score (%)')
    ax.set_title('Score distribution per method (all subjects/exercises)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'score_distribution_boxplot.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot three-methods comparison results")
    parser.add_argument('csv_path', help="Path to three_methods_comparison.csv")
    parser.add_argument('--outdir', default=None,
                         help="Where to save PNGs (default: a 'comparison_plots' folder "
                              "next to the CSV)")
    args = parser.parse_args()

    csv_dir = os.path.dirname(os.path.abspath(args.csv_path)) or '.'
    outdir = args.outdir or os.path.join(csv_dir, 'comparison_plots')
    os.makedirs(outdir, exist_ok=True)

    df = load_data(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}\n")

    plot_grouped_bars(df, outdir)
    plot_heatmaps(df, outdir)
    plot_agreement(df, outdir)
    plot_confusion_matrices(df, outdir)
    plot_boxplot(df, outdir)

    print(f"\nAll figures saved to: {outdir}")


if __name__ == '__main__':
    main()