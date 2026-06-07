import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a strong CNN ensemble for Kaggle Digit Recognizer."
    )
    parser.add_argument("--train-csv", default="train/train.csv")
    parser.add_argument("--test-csv", default="test/test.csv")
    parser.add_argument("--output", default="test/submission_high_score.csv")
    parser.add_argument("--models", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--tta", type=int, default=4, help="Number of augmented test predictions.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Use a tiny subset to verify the pipeline quickly.",
    )
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(train_csv, test_csv, quick_check=False):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if quick_check:
        train_df = train_df.sample(n=min(3000, len(train_df)), random_state=7)
        test_df = test_df.head(min(512, len(test_df)))

    y = train_df["label"].astype("int64").to_numpy()
    x = train_df.drop(columns=["label"]).to_numpy(dtype="float32").reshape(-1, 28, 28, 1)
    test_x = test_df.to_numpy(dtype="float32").reshape(-1, 28, 28, 1)

    x = x / 255.0
    test_x = test_x / 255.0
    return x, y, test_x


def make_augmenter(seed):
    return keras.Sequential(
        [
            layers.RandomRotation(0.08, fill_mode="constant", seed=seed + 1),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", seed=seed + 2),
            layers.RandomZoom((-0.10, 0.10), (-0.10, 0.10), fill_mode="constant", seed=seed + 3),
        ],
        name="digit_augmentation",
    )


def conv_block(x, filters, dropout):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_classifier():
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = conv_block(inputs, 32, 0.20)
    x = conv_block(x, 64, 0.25)
    x = conv_block(x, 128, 0.30)
    x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="mnist_cnn_classifier")


def build_training_model(seed):
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = make_augmenter(seed)(inputs)
    outputs = build_classifier()(x)
    model = keras.Model(inputs, outputs, name="mnist_cnn_with_augmentation")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_with_tta(model, test_x, tta_rounds, seed, batch_size):
    probs = model.predict(test_x, batch_size=batch_size, verbose=0)

    if tta_rounds <= 0:
        return probs

    augmenter = make_augmenter(seed + 10000)
    classifier = model.get_layer("mnist_cnn_classifier")
    test_tensor = tf.convert_to_tensor(test_x, dtype=tf.float32)

    for _ in range(tta_rounds):
        augmented = augmenter(test_tensor, training=True)
        probs += classifier.predict(augmented, batch_size=batch_size, verbose=0)

    return probs / (tta_rounds + 1)


def train_one_model(x, y, test_x, args, model_index):
    seed = args.seed + model_index * 97
    set_seed(seed)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.val_size,
        random_state=seed,
        stratify=y,
    )

    model = build_training_model(seed)
    checkpoint_path = Path("models") / f"digit_cnn_model_{model_index + 1}.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
    ]

    print(f"\nTraining model {model_index + 1}/{args.models} with seed {seed}")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    best_val = max(history.history["val_accuracy"])
    model = keras.models.load_model(checkpoint_path)
    probs = predict_with_tta(model, test_x, args.tta, seed, args.batch_size)
    print(f"Best validation accuracy for model {model_index + 1}: {best_val:.5f}")
    return probs, best_val


def save_submission(probs, output_path):
    labels = np.argmax(probs, axis=1)
    submission = pd.DataFrame(
        {
            "ImageId": np.arange(1, len(labels) + 1),
            "Label": labels.astype(int),
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def main():
    args = parse_args()
    set_seed(args.seed)

    print("Loading data...")
    x, y, test_x = load_data(args.train_csv, args.test_csv, args.quick_check)
    print(f"Train images: {len(x):,}, test images: {len(test_x):,}")

    ensemble_probs = np.zeros((len(test_x), 10), dtype="float64")
    val_scores = []
    for model_index in range(args.models):
        probs, val_acc = train_one_model(x, y, test_x, args, model_index)
        ensemble_probs += probs
        val_scores.append(val_acc)

    ensemble_probs /= args.models
    output_path = save_submission(ensemble_probs, args.output)

    print("\nDone.")
    print(f"Validation accuracies: {[round(score, 5) for score in val_scores]}")
    print(f"Mean validation accuracy: {np.mean(val_scores):.5f}")
    print(f"Submission saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
