# MNIST Deep Learning Assignment

This project trains three TensorFlow/Keras neural networks on the MNIST handwritten digit dataset:

1. A simple baseline model.
2. A deliberately overfitted model.
3. An improved model that reduces overfitting without shrinking the large architecture.

The project runs end-to-end with:

```bash
python main.py
```

## What MNIST Is

MNIST is a classic dataset of handwritten digit images. Each image is a 28 by 28 grayscale picture of one digit from 0 to 9. The dataset is commonly used to practice image classification because it is small, clean, and easy to load from `tensorflow.keras.datasets.mnist`.

This project uses the original MNIST split:

- 60,000 original training images
- 10,000 original test images

The 60,000 original training images are split once and kept fixed:

- First 50,000 images: training set
- Last 10,000 images: validation set
- Original 10,000 test images: unchanged test set

All pixel values are normalized from the original range `[0, 255]` to `[0, 1]`.

## What Overfitting Is

Overfitting happens when a model learns the training data too closely instead of learning patterns that generalize well to new data. An overfitted model usually has very high training accuracy but lower validation or test accuracy. Its validation loss may stop improving or increase while training loss continues to decrease.

## Baseline Model Design

The baseline model is carefully tuned so it can act as a stable reference without massively overfitting:

- Flatten
- Dense(96, ReLU)
- Dropout, rate `0.4`
- Dense(10, Softmax)

It relies entirely on Dropout for regularization and is trained for a maximum of 60 epochs (with EarlyStopping usually halting it around epoch 35-50). This keeps the baseline optimally balanced, preventing extreme divergence in the loss ratio.

## How Overfitting Was Intentionally Created

The overfitted model uses a much larger architecture than the baseline:

- Flatten
- Dense(1024, ReLU)
- Dense(1024, ReLU)
- Dense(512, ReLU)
- Dense(512, ReLU)
- Dense(256, ReLU)
- Dense(10, Softmax)

No Dropout, BatchNormalization, or L2 regularization is used. The model is trained for 60 epochs, which gives it enough time and capacity to memorize training examples and show overfitting behavior in the accuracy and loss plots.

## How Overfitting Was Reduced

The improved model keeps the same large Dense layer sizes as the overfitted model. It does not remove layers or reduce neurons. Overfitting is reduced only by changing the training strategy:

- Data Augmentation (RandomRotation, RandomTranslation, RandomZoom) built directly into the preprocessing pipeline
- Dropout layers with rate `0.3`
- L2 kernel regularization on Dense layers with strength `0.0002`
- EarlyStopping with `restore_best_weights=True`
- ReduceLROnPlateau to lower the learning rate when validation loss stops improving

By combining these techniques, the improved model perfectly eliminates the massive overfitting problem while maintaining the exact same number of parameters as the overfitted model.

The same training, validation, and test split is used for all three models.

## Project Structure

```text
mnist_assignment/
|
├── main.py
├── models.py
├── train.py
├── utils.py
├── plots/
├── saved_models/
├── requirements.txt
└── README.md
```

## Output Files

After running the project, the following model files are saved:

- `saved_models/baseline_model.keras`
- `saved_models/overfit_model.keras`
- `saved_models/improved_model.keras`

The following plots are saved:

- `plots/baseline_model_accuracy.png`
- `plots/baseline_model_loss.png`
- `plots/overfit_model_accuracy.png`
- `plots/overfit_model_loss.png`
- `plots/improved_model_accuracy.png`
- `plots/improved_model_loss.png`
- `plots/model_accuracy_comparison.png`
- `plots/model_epochs_comparison.png`

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full assignment:

```bash
python main.py
```

The script automatically loads MNIST, creates the fixed split, trains all three models, saves plots, saves trained models, prints evaluation metrics, and prints the final comparison table.
