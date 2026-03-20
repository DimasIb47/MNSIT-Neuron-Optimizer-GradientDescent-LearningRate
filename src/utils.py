"""
Utility functions untuk plotting dan saving hasil eksperimen MNIST DNN.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ============================================================
# Konfigurasi style global
# ============================================================
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'legend.fontsize': 10,
    'figure.dpi': 120,
})

# Palet warna yang konsisten
COLORS = ['#58a6ff', '#f0883e', '#3fb950', '#d2a8ff', '#ff7b72',
           '#79c0ff', '#ffa657', '#56d364', '#bc8cff', '#ffa198']


def ensure_dir(path):
    """Buat direktori kalau belum ada."""
    os.makedirs(path, exist_ok=True)


def plot_training_curves(history, title, save_path):
    """
    Plot kurva loss dan accuracy selama training.

    Args:
        history: dict dengan keys 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        title: judul grafik
        save_path: path untuk menyimpan gambar
    """
    ensure_dir(os.path.dirname(save_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['loss']) + 1)

    # --- Loss ---
    ax1.plot(epochs, history['loss'], color=COLORS[0], linewidth=2,
             marker='o', markersize=4, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], color=COLORS[1], linewidth=2,
             marker='s', markersize=4, label='Validation Loss', linestyle='--')
    ax1.set_title(f'Loss — {title}', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Accuracy ---
    ax2.plot(epochs, history['accuracy'], color=COLORS[2], linewidth=2,
             marker='o', markersize=4, label='Training Accuracy')
    ax2.plot(epochs, history['val_accuracy'], color=COLORS[3], linewidth=2,
             marker='s', markersize=4, label='Validation Accuracy', linestyle='--')
    ax2.set_title(f'Accuracy — {title}', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Grafik disimpan: {save_path}")


def plot_comparison_bar(results_df, x_col, y_col, title, save_path,
                        hue_col=None, ylabel=None, ylim=None):
    """
    Bar chart perbandingan.

    Args:
        results_df: DataFrame dengan hasil
        x_col: kolom untuk sumbu X
        y_col: kolom untuk sumbu Y
        title: judul grafik
        save_path: path untuk menyimpan gambar
        hue_col: kolom untuk warna grouping (optional)
        ylabel: label sumbu Y (optional)
        ylim: tuple (min, max) untuk sumbu Y (optional)
    """
    ensure_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(10, 6))

    if hue_col and hue_col in results_df.columns:
        groups = results_df[hue_col].unique()
        n_groups = len(groups)
        x_labels = results_df[x_col].unique()
        x_indices = np.arange(len(x_labels))
        bar_width = 0.8 / n_groups

        for i, group in enumerate(groups):
            group_data = results_df[results_df[hue_col] == group]
            # Align by x_labels order
            vals = []
            for xl in x_labels:
                match = group_data[group_data[x_col] == xl]
                vals.append(match[y_col].values[0] if len(match) > 0 else 0)
            bars = ax.bar(x_indices + i * bar_width, vals,
                          bar_width, label=str(group), color=COLORS[i % len(COLORS)],
                          edgecolor='white', linewidth=0.5)
            # Label di atas bar
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8,
                        color='#c9d1d9')

        ax.set_xticks(x_indices + bar_width * (n_groups - 1) / 2)
        ax.set_xticklabels(x_labels, rotation=30, ha='right')
        ax.legend(title=hue_col)
    else:
        x_labels = results_df[x_col].astype(str).values
        vals = results_df[y_col].values
        bars = ax.bar(range(len(x_labels)), vals, color=COLORS[:len(vals)],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9,
                    color='#c9d1d9')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=30, ha='right')

    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel or y_col)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Grafik disimpan: {save_path}")


def plot_lr_heatmap(results_df, save_path):
    """
    Heatmap learning rate × (optimizer + GD variant) → test accuracy.

    Args:
        results_df: DataFrame dengan kolom 'learning_rate', 'optimizer',
                     'gd_variant', 'test_accuracy'
        save_path: path untuk menyimpan gambar
    """
    ensure_dir(os.path.dirname(save_path))

    # Buat label gabungan
    df = results_df.copy()
    df['config'] = df['optimizer'] + ' + ' + df['gd_variant']

    pivot = df.pivot_table(values='test_accuracy', index='config',
                           columns='learning_rate', aggfunc='first')

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, cmap='YlGnBu', aspect='auto')

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate setiap cell
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = 'black' if val > 0.9 else 'white'
                ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                        fontsize=8, color=text_color, fontweight='bold')

    ax.set_title('Heatmap: Test Accuracy × Learning Rate', fontweight='bold')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Konfigurasi (Optimizer + GD Variant)')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Test Accuracy', color='#c9d1d9')
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8b949e')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Heatmap disimpan: {save_path}")


def plot_training_time_bar(results_df, save_path):
    """
    Bar chart waktu training per GD variant × optimizer.
    """
    ensure_dir(os.path.dirname(save_path))

    fig, ax = plt.subplots(figsize=(10, 6))

    df = results_df.copy()
    df['config'] = df['gd_variant'] + '\n(' + df['optimizer'] + ')'

    bars = ax.bar(range(len(df)), df['training_time_s'].values,
                  color=COLORS[:len(df)], edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, df['training_time_s'].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9,
                color='#c9d1d9')

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['config'].values, rotation=0, ha='center')
    ax.set_title('Waktu Training per Konfigurasi GD × Optimizer', fontweight='bold')
    ax.set_xlabel('Konfigurasi')
    ax.set_ylabel('Waktu (detik)')
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Grafik disimpan: {save_path}")


def save_results_csv(results_df, save_path):
    """Simpan DataFrame hasil ke CSV."""
    ensure_dir(os.path.dirname(save_path))
    results_df.to_csv(save_path, index=False)
    print(f"  [✓] CSV disimpan: {save_path}")


def print_table(df, title=""):
    """Print DataFrame sebagai tabel yang rapi di terminal."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(df.to_string(index=False))
    print()
