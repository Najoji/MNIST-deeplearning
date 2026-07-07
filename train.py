"""Training helpers for the MNIST assignment."""

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils import SAVED_MODELS_DIR, plot_training_history, print_metrics


def compile_model(model, learning_rate=0.001):
    """Compile a model for sparse-label digit classification."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(
    model,
    model_key,
    label,
    train_data,
    val_data,
    test_data,
    epochs,
    batch_size=128,
    callbacks=None,
):
    """Train, plot, evaluate, save, and return metrics for one model."""
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data

    print(f"\nTraining {label}...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_history(history, model_key)

    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    metrics = {
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
    }

    print_metrics(label, metrics)
    model.save(SAVED_MODELS_DIR / f"{model_key}.keras")

    return metrics


def baseline_model_callbacks():
    """Callbacks used to train the baseline until validation accuracy plateaus."""
    return [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def improved_model_callbacks():
    """Callbacks used only for the regularized improved model."""
    return [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
