"""Model definitions for the MNIST assignment."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.regularizers import l2


INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10


def build_baseline_model():
    """Create the simple feed-forward neural network for Part 1."""
    model = Sequential(
        [
            Input(shape=INPUT_SHAPE, name="input"),
            Flatten(name="flatten"),
            Dense(128, activation="relu", name="dense_128"),
            Dense(64, activation="relu", name="dense_64"),
            Dense(NUM_CLASSES, activation="softmax", name="output"),
        ],
        name="baseline_model",
    )
    return model


def build_overfit_model():
    """Create a deliberately high-capacity model with no regularization."""
    model = Sequential(
        [
            Input(shape=INPUT_SHAPE, name="input"),
            Flatten(name="flatten"),
            Dense(1024, activation="relu", name="dense_1024_a"),
            Dense(1024, activation="relu", name="dense_1024_b"),
            Dense(512, activation="relu", name="dense_512_a"),
            Dense(512, activation="relu", name="dense_512_b"),
            Dense(256, activation="relu", name="dense_256"),
            Dense(NUM_CLASSES, activation="softmax", name="output"),
        ],
        name="overfit_model",
    )
    return model


def build_improved_model(l2_strength=0.001, dropout_rate=0.5):
    """
    Create the same high-capacity architecture as the overfit model, but with
    regularization added to reduce overfitting.

    The Dense layer sizes are unchanged from build_overfit_model().
    """
    regularizer = l2(l2_strength)

    model = Sequential(
        [
            Input(shape=INPUT_SHAPE, name="input"),
            Flatten(name="flatten"),
            Dense(
                1024,
                activation="relu",
                kernel_regularizer=regularizer,
                name="dense_1024_a",
            ),
            Dropout(dropout_rate, name="dropout_1024_a"),
            Dense(
                1024,
                activation="relu",
                kernel_regularizer=regularizer,
                name="dense_1024_b",
            ),
            Dropout(dropout_rate, name="dropout_1024_b"),
            Dense(
                512,
                activation="relu",
                kernel_regularizer=regularizer,
                name="dense_512_a",
            ),
            Dropout(dropout_rate, name="dropout_512_a"),
            Dense(
                512,
                activation="relu",
                kernel_regularizer=regularizer,
                name="dense_512_b",
            ),
            Dropout(dropout_rate, name="dropout_512_b"),
            Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizer,
                name="dense_256",
            ),
            Dropout(dropout_rate, name="dropout_256"),
            Dense(NUM_CLASSES, activation="softmax", name="output"),
        ],
        name="improved_model",
    )
    return model
