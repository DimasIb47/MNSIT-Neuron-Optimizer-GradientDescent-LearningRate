"""
Tugas 1 - Pengantar Deep Learning
MNIST Deep Neural Network — Eksperimen Lengkap

Eksperimen:
  0. Konfigurasi jumlah neuron hidden layer
  1. Optimizer: RMSprop vs Adam
  2. Gradient Descent variant: SGD, Batch, Mini-batch
  3. Learning Rate optimization
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import utils dari folder yang sama
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    plot_training_curves,
    plot_comparison_bar,
    plot_lr_heatmap,
    plot_training_time_bar,
    save_results_csv,
    print_table,
    ensure_dir,
)

# ============================================================
# KONFIGURASI
# ============================================================
EPOCHS = 15
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'results')

# Konfigurasi neuron (Hidden Layer 1, Hidden Layer 2)
NEURON_CONFIGS = {
    'A (64,32)':   (64, 32),
    'B (128,64)':  (128, 64),
    'C (256,128)': (256, 128),
    'D (512,256)': (512, 256),
    'E (128,128)': (128, 128),
}

# Learning rates untuk eksperimen
LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

# Default learning rate untuk eksperimen non-LR
DEFAULT_LR = 0.001


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
def load_data():
    """Load dan preprocess MNIST dataset."""
    print("\n" + "=" * 60)
    print("  LOADING MNIST DATASET")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Flatten gambar 28x28 → 784 dan normalisasi ke 0-1
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    # One-hot encoding label
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print(f"  Training data : {x_train.shape[0]} gambar, shape={x_train.shape}")
    print(f"  Testing data  : {x_test.shape[0]} gambar, shape={x_test.shape}")
    print(f"  Labels        : one-hot encoded, 10 kelas")

    return (x_train, y_train), (x_test, y_test)


# ============================================================
# MODEL BUILDER
# ============================================================
def build_model(hidden1, hidden2, optimizer):
    """
    Bangun DNN dengan 2 hidden layer.

    Args:
        hidden1: jumlah neuron hidden layer 1
        hidden2: jumlah neuron hidden layer 2
        optimizer: keras optimizer instance

    Returns:
        compiled keras model
    """
    model = keras.Sequential([
        layers.Dense(hidden1, activation='relu', input_shape=(784,),
                     name='hidden_layer_1'),
        layers.Dense(hidden2, activation='relu',
                     name='hidden_layer_2'),
        layers.Dense(10, activation='softmax',
                     name='output_layer'),
    ])

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def create_optimizer(name, learning_rate):
    """Buat optimizer berdasarkan nama dan learning rate."""
    if name.lower() == 'rmsprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif name.lower() == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer tidak dikenal: {name}")


def train_model(model, x_train, y_train, x_test, y_test,
                batch_size, epochs=EPOCHS, verbose=1):
    """
    Train model dan return history + waktu training.

    Returns:
        history_dict: dict dengan loss, val_loss, accuracy, val_accuracy
        train_time: waktu training dalam detik
    """
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=verbose,
    )

    train_time = time.time() - start_time

    return history.history, train_time


# ============================================================
# EKSPERIMEN 0: Konfigurasi Neuron
# ============================================================
def experiment_neurons(x_train, y_train, x_test, y_test):
    """Cari konfigurasi neuron terbaik."""
    print("\n" + "=" * 60)
    print("  EKSPERIMEN 0: Konfigurasi Jumlah Neuron")
    print("  Optimizer: Adam (default), LR: 0.001, Batch: 32")
    print("=" * 60)

    results = []

    for name, (h1, h2) in NEURON_CONFIGS.items():
        print(f"\n--- Config {name} ---")
        optimizer = create_optimizer('adam', DEFAULT_LR)
        model = build_model(h1, h2, optimizer)
        history, train_time = train_model(
            model, x_train, y_train, x_test, y_test,
            batch_size=32, verbose=0
        )

        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        results.append({
            'config': name,
            'hidden1': h1,
            'hidden2': h2,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'training_time_s': train_time,
        })

        # Simpan training curves
        plot_training_curves(
            history, f'Neuron Config {name}',
            os.path.join(RESULTS_DIR, 'exp0_neurons', f'curves_{name.split()[0]}.png')
        )

        print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {train_time:.1f}s")

    df = pd.DataFrame(results)
    save_results_csv(df, os.path.join(RESULTS_DIR, 'exp0_neurons', 'results.csv'))
    print_table(df[['config', 'train_accuracy', 'test_accuracy',
                     'train_loss', 'test_loss', 'training_time_s']],
                "Hasil Eksperimen Neuron")

    # Bar chart perbandingan
    plot_comparison_bar(
        df, 'config', 'test_accuracy',
        'Perbandingan Test Accuracy per Konfigurasi Neuron',
        os.path.join(RESULTS_DIR, 'exp0_neurons', 'comparison_accuracy.png'),
        ylabel='Test Accuracy'
    )

    # Cari config terbaik
    best_idx = df['test_accuracy'].idxmax()
    best_config = df.loc[best_idx]
    best_h1 = int(best_config['hidden1'])
    best_h2 = int(best_config['hidden2'])
    print(f"\n  ★ Konfigurasi terbaik: {best_config['config']}")
    print(f"    Hidden Layer 1: {best_h1}, Hidden Layer 2: {best_h2}")
    print(f"    Test Accuracy: {best_config['test_accuracy']:.4f}")

    return best_h1, best_h2, df


# ============================================================
# EKSPERIMEN 1: Optimizer (RMSprop vs Adam)
# ============================================================
def experiment_optimizers(x_train, y_train, x_test, y_test, h1, h2):
    """Bandingkan RMSprop vs Adam."""
    print("\n" + "=" * 60)
    print(f"  EKSPERIMEN 1: Optimizer (RMSprop vs Adam)")
    print(f"  Neuron: ({h1},{h2}), LR: {DEFAULT_LR}, Batch: 32")
    print("=" * 60)

    results = []

    for opt_name in ['RMSprop', 'Adam']:
        print(f"\n--- Optimizer: {opt_name} ---")
        optimizer = create_optimizer(opt_name, DEFAULT_LR)
        model = build_model(h1, h2, optimizer)
        history, train_time = train_model(
            model, x_train, y_train, x_test, y_test,
            batch_size=32, verbose=0
        )

        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        results.append({
            'optimizer': opt_name,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'training_time_s': train_time,
        })

        plot_training_curves(
            history, f'Optimizer: {opt_name}',
            os.path.join(RESULTS_DIR, 'exp1_optimizer', f'curves_{opt_name}.png')
        )

        print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {train_time:.1f}s")

    df = pd.DataFrame(results)
    save_results_csv(df, os.path.join(RESULTS_DIR, 'exp1_optimizer', 'results.csv'))
    print_table(df, "Hasil Eksperimen Optimizer")

    plot_comparison_bar(
        df, 'optimizer', 'test_accuracy',
        'Perbandingan Test Accuracy: RMSprop vs Adam',
        os.path.join(RESULTS_DIR, 'exp1_optimizer', 'comparison_accuracy.png'),
        ylabel='Test Accuracy'
    )

    return df


# ============================================================
# EKSPERIMEN 2: Gradient Descent Variant
# ============================================================
def experiment_gd_variants(x_train, y_train, x_test, y_test, h1, h2):
    """Bandingkan SGD, Batch GD, dan Mini-batch GD dengan kedua optimizer."""
    print("\n" + "=" * 60)
    print(f"  EKSPERIMEN 2: Gradient Descent Variants")
    print(f"  Neuron: ({h1},{h2}), LR: {DEFAULT_LR}")
    print("=" * 60)

    n_train = x_train.shape[0]  # 60000

    gd_configs = {
        'Stochastic (batch=1)':   1,
        'Mini-Batch (batch=32)':  32,
        'Batch (batch=all)':      n_train,
    }

    results = []

    for opt_name in ['RMSprop', 'Adam']:
        for gd_name, batch_size in gd_configs.items():
            print(f"\n--- {opt_name} + {gd_name} ---")

            # Untuk batch GD dan SGD, kurangi epoch atau sesuaikan verbose
            # SGD dengan batch=1 akan sangat lambat, jadi kita kurangi epoch
            actual_epochs = EPOCHS
            if batch_size == 1:
                actual_epochs = 1  # SGD terlalu lambat untuk 15 epoch, jadi 1 saja
                print(f"  ⚠ SGD batch=1 sangat lambat, hanya {actual_epochs} epoch")

            optimizer = create_optimizer(opt_name, DEFAULT_LR)
            model = build_model(h1, h2, optimizer)
            history, train_time = train_model(
                model, x_train, y_train, x_test, y_test,
                batch_size=batch_size, epochs=actual_epochs, verbose=1
            )

            train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

            results.append({
                'optimizer': opt_name,
                'gd_variant': gd_name,
                'batch_size': batch_size,
                'epochs': actual_epochs,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'training_time_s': round(train_time, 1),
            })

            safe_name = gd_name.split('(')[0].strip().replace(' ', '_')
            plot_training_curves(
                history, f'{opt_name} + {gd_name}',
                os.path.join(RESULTS_DIR, 'exp2_gd_variant',
                             f'curves_{opt_name}_{safe_name}.png')
            )

            print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
                  f"Time: {train_time:.1f}s | Epochs: {actual_epochs}")

    df = pd.DataFrame(results)
    save_results_csv(df, os.path.join(RESULTS_DIR, 'exp2_gd_variant', 'results.csv'))
    print_table(df[['optimizer', 'gd_variant', 'batch_size', 'epochs',
                     'train_accuracy', 'test_accuracy', 'training_time_s']],
                "Hasil Eksperimen GD Variant")

    # Bar chart grouped by optimizer
    plot_comparison_bar(
        df, 'gd_variant', 'test_accuracy',
        'Perbandingan Test Accuracy per GD Variant',
        os.path.join(RESULTS_DIR, 'exp2_gd_variant', 'comparison_accuracy.png'),
        hue_col='optimizer', ylabel='Test Accuracy'
    )

    # Training time comparison
    plot_training_time_bar(
        df, os.path.join(RESULTS_DIR, 'exp2_gd_variant', 'comparison_time.png')
    )

    return df


# ============================================================
# EKSPERIMEN 3: Learning Rate
# ============================================================
def experiment_learning_rate(x_train, y_train, x_test, y_test, h1, h2):
    """Variasi learning rate untuk semua kombinasi optimizer × GD variant."""
    print("\n" + "=" * 60)
    print(f"  EKSPERIMEN 3: Learning Rate Optimization")
    print(f"  Neuron: ({h1},{h2})")
    print(f"  LR values: {LEARNING_RATES}")
    print("=" * 60)

    n_train = x_train.shape[0]

    gd_configs = {
        'Stochastic':  1,
        'Mini-Batch':  32,
        'Batch':       n_train,
    }

    results = []
    total_experiments = len(['RMSprop', 'Adam']) * len(gd_configs) * len(LEARNING_RATES)
    current = 0

    for opt_name in ['RMSprop', 'Adam']:
        for gd_name, batch_size in gd_configs.items():
            for lr in LEARNING_RATES:
                current += 1
                print(f"\n[{current}/{total_experiments}] "
                      f"{opt_name} + {gd_name} + LR={lr}")

                actual_epochs = EPOCHS
                if batch_size == 1:
                    actual_epochs = 1
                    print(f"  ⚠ SGD: hanya {actual_epochs} epoch")

                try:
                    optimizer = create_optimizer(opt_name, lr)
                    model = build_model(h1, h2, optimizer)
                    history, train_time = train_model(
                        model, x_train, y_train, x_test, y_test,
                        batch_size=batch_size, epochs=actual_epochs, verbose=1
                    )

                    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
                    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    train_acc = test_acc = 0.0
                    train_loss = test_loss = float('inf')
                    train_time = 0.0

                results.append({
                    'optimizer': opt_name,
                    'gd_variant': gd_name,
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'epochs': actual_epochs,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'training_time_s': round(train_time, 1),
                })

                print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    df = pd.DataFrame(results)
    save_results_csv(df, os.path.join(RESULTS_DIR, 'exp3_learning_rate', 'results.csv'))

    # Heatmap
    plot_lr_heatmap(df, os.path.join(RESULTS_DIR, 'exp3_learning_rate', 'heatmap.png'))

    # Cari LR terbaik per optimizer × GD variant
    best_lr_results = []
    for opt_name in ['RMSprop', 'Adam']:
        for gd_name in gd_configs.keys():
            subset = df[(df['optimizer'] == opt_name) & (df['gd_variant'] == gd_name)]
            if len(subset) > 0:
                best_idx = subset['test_accuracy'].idxmax()
                best = subset.loc[best_idx]
                best_lr_results.append({
                    'optimizer': opt_name,
                    'gd_variant': gd_name,
                    'best_learning_rate': best['learning_rate'],
                    'test_accuracy': best['test_accuracy'],
                    'test_loss': best['test_loss'],
                })

    best_df = pd.DataFrame(best_lr_results)
    save_results_csv(best_df, os.path.join(RESULTS_DIR, 'exp3_learning_rate',
                                            'best_lr.csv'))
    print_table(best_df, "Learning Rate Terbaik per Konfigurasi")

    return df, best_df


# ============================================================
# TABEL RINGKASAN AKHIR
# ============================================================
def create_final_summary(neuron_df, optimizer_df, gd_df, lr_df, best_lr_df, results_dir):
    """Buat tabel ringkasan akhir dari semua eksperimen."""
    print("\n" + "=" * 60)
    print("  RINGKASAN AKHIR SEMUA EKSPERIMEN")
    print("=" * 60)

    # Tabel ringkasan gabungan
    summary_rows = []

    # Dari eksperimen neuron
    best_neuron = neuron_df.loc[neuron_df['test_accuracy'].idxmax()]
    summary_rows.append({
        'Eksperimen': 'Neuron Terbaik',
        'Konfigurasi': best_neuron['config'],
        'Test Accuracy': f"{best_neuron['test_accuracy']:.4f}",
        'Test Loss': f"{best_neuron['test_loss']:.4f}",
    })

    # Dari eksperimen optimizer
    best_opt = optimizer_df.loc[optimizer_df['test_accuracy'].idxmax()]
    summary_rows.append({
        'Eksperimen': 'Optimizer Terbaik',
        'Konfigurasi': best_opt['optimizer'],
        'Test Accuracy': f"{best_opt['test_accuracy']:.4f}",
        'Test Loss': f"{best_opt['test_loss']:.4f}",
    })

    # Dari eksperimen GD variant
    best_gd = gd_df.loc[gd_df['test_accuracy'].idxmax()]
    summary_rows.append({
        'Eksperimen': 'GD Variant Terbaik',
        'Konfigurasi': f"{best_gd['optimizer']} + {best_gd['gd_variant']}",
        'Test Accuracy': f"{best_gd['test_accuracy']:.4f}",
        'Test Loss': f"{best_gd['test_loss']:.4f}",
    })

    # Dari eksperimen LR (overall best)
    best_lr_row = best_lr_df.loc[best_lr_df['test_accuracy'].idxmax()]
    summary_rows.append({
        'Eksperimen': 'Learning Rate Terbaik',
        'Konfigurasi': (f"{best_lr_row['optimizer']} + "
                        f"{best_lr_row['gd_variant']} + "
                        f"LR={best_lr_row['best_learning_rate']}"),
        'Test Accuracy': f"{best_lr_row['test_accuracy']:.4f}",
        'Test Loss': f"{best_lr_row['test_loss']:.4f}",
    })

    summary_df = pd.DataFrame(summary_rows)
    save_results_csv(summary_df, os.path.join(results_dir, 'final_summary.csv'))
    print_table(summary_df, "★ RINGKASAN AKHIR ★")

    return summary_df


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "★" * 60)
    print("  TUGAS 1 — PENGANTAR DEEP LEARNING")
    print("  MNIST Deep Neural Network Experiments")
    print("★" * 60)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    ensure_dir(RESULTS_DIR)

    # ── Memuat Hasil Sebelumnya (Exp 0 & 1) ──
    print("\n  [INFO] Memuat ulang hasil Eksperimen 0 & 1 dari CSV...")
    neuron_df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp0_neurons', 'results.csv'))
    best_h1 = 512
    best_h2 = 256
    
    optimizer_df = pd.read_csv(os.path.join(RESULTS_DIR, 'exp1_optimizer', 'results.csv'))

    # ── Eksperimen 2: GD Variants ──
    gd_df = experiment_gd_variants(
        x_train, y_train, x_test, y_test, best_h1, best_h2
    )

    # ── Eksperimen 3: Learning Rate ──
    lr_df, best_lr_df = experiment_learning_rate(
        x_train, y_train, x_test, y_test, best_h1, best_h2
    )

    # ── Ringkasan Akhir ──
    summary_df = create_final_summary(
        neuron_df, optimizer_df, gd_df, lr_df, best_lr_df, RESULTS_DIR
    )

    print("\n" + "★" * 60)
    print("  SEMUA EKSPERIMEN SELESAI!")
    print(f"  Hasil disimpan di: {RESULTS_DIR}")
    print("★" * 60)


if __name__ == '__main__':
    main()
