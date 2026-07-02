"""Utility functions for loading data, plotting, and reporting results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
TRAIN_SIZE = 50000
VALIDATION_SIZE = 10000


def ensure_directories():
    """Create output directories used by the project."""
    PLOTS_DIR.mkdir(exist_ok=True)
    SAVED_MODELS_DIR.mkdir(exist_ok=True)


def load_mnist_fixed_split():
    """
    Load MNIST, normalize pixel values, and create the required fixed split.

    Original MNIST training data has 60,000 images. The first 50,000 are used
    for training and the final 10,000 are used for validation. The official
    10,000-image test set is kept unchanged.
    """
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train_full[:TRAIN_SIZE]
    y_train = y_train_full[:TRAIN_SIZE]
    x_val = x_train_full[TRAIN_SIZE : TRAIN_SIZE + VALIDATION_SIZE]
    y_val = y_train_full[TRAIN_SIZE : TRAIN_SIZE + VALIDATION_SIZE]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def plot_training_history(history, model_name):
    """Save accuracy and loss curves for a trained model."""
    epochs = np.arange(1, len(history.history["accuracy"]) + 1)
    readable_name = model_name.replace("_", " ").title()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{readable_name}: Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"], label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
    plt.title(f"{readable_name}: Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name}_loss.png", dpi=150)
    plt.close()


def print_metrics(label, metrics):
    """Print a compact summary for one model."""
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Training accuracy:   {metrics['train_accuracy']:.4f}")
    print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Test accuracy:       {metrics['test_accuracy']:.4f}")
    print(f"Final training loss: {metrics['train_loss']:.4f}")
    print(f"Final val loss:      {metrics['val_loss']:.4f}")
    print(f"Test loss:           {metrics['test_loss']:.4f}")


def print_comparison_table(results):
    """Print the final comparison table required by the assignment."""
    print("\nModel Comparison")
    print("=" * 78)
    print(
        f"{'Model':<20} | {'Train Accuracy':>15} | "
        f"{'Validation Accuracy':>19} | {'Test Accuracy':>13}"
    )
    print("-" * 78)

    for model_name, metrics in results.items():
        print(
            f"{model_name:<20} | "
            f"{metrics['train_accuracy']:>15.4f} | "
            f"{metrics['val_accuracy']:>19.4f} | "
            f"{metrics['test_accuracy']:>13.4f}"
        )
    print("=" * 78)
