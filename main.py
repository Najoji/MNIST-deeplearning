"""Run the complete MNIST deep learning assignment end-to-end."""

import random

import numpy as np
import tensorflow as tf

from models import build_baseline_model, build_improved_model, build_overfit_model
from train import (
    compile_model,
    improved_model_callbacks,
    train_and_evaluate,
)
from utils import ensure_directories, load_mnist_fixed_split, print_comparison_table


SEED = 42


def set_reproducibility(seed=SEED):
    """Set random seeds so repeated runs are as consistent as possible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    set_reproducibility()
    ensure_directories()

    train_data, val_data, test_data = load_mnist_fixed_split()

    print("MNIST fixed split")
    print("-----------------")
    print(f"Training images:   {train_data[0].shape[0]}")
    print(f"Validation images: {val_data[0].shape[0]}")
    print(f"Test images:       {test_data[0].shape[0]}")

    results = {}

    set_reproducibility()
    baseline_model = compile_model(build_baseline_model())
    results["Baseline Model"] = train_and_evaluate(
        baseline_model,
        model_key="baseline_model",
        label="Baseline Model",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=15,
        batch_size=128,
    )

    set_reproducibility()
    overfit_model = compile_model(build_overfit_model())
    results["Overfitted Model"] = train_and_evaluate(
        overfit_model,
        model_key="overfit_model",
        label="Overfitted Model",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=60,
        batch_size=128,
    )

    set_reproducibility()
    improved_model = compile_model(build_improved_model())
    results["Improved Model"] = train_and_evaluate(
        improved_model,
        model_key="improved_model",
        label="Improved Model",
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        epochs=35,
        batch_size=128,
        callbacks=improved_model_callbacks(),
    )

    print_comparison_table(results)
    print("\nSaved models are in the saved_models/ folder.")
    print("Training plots are in the plots/ folder.")


if __name__ == "__main__":
    main()
