"""Train the original MNIST MLPs with a separate validation split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "comparison", "deep"), default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--limit-train", type=int, default=None, help="Optional quick-demo subset")
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.validation_fraction < 1:
        parser.error("epochs/batch-size must be positive; validation-fraction must be in (0, 1)")
    if any(n is not None and n < 1 for n in (args.limit_train, args.limit_validation)):
        parser.error("Subset limits must be positive")
    return args


def split_indices(labels, fraction, seed, np):
    """Split each class independently; do not use the official test set."""
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for class_id in range(10):
        indices = rng.permutation(np.flatnonzero(labels == class_id))
        count = max(1, int(round(len(indices) * fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return rng.permutation(train), rng.permutation(validation)


def build_model(tf, deep=False):
    if deep:
        layers = [tf.keras.Input(shape=(784,)),
                  tf.keras.layers.Dense(1024, activation="relu"),
                  tf.keras.layers.Dense(512, activation="relu"),
                  tf.keras.layers.Dense(512, activation="relu"),
                  tf.keras.layers.Dense(10, activation="softmax")]
    else:
        layers = [tf.keras.Input(shape=(784,)),
                  tf.keras.layers.Dense(512, activation="tanh"),
                  tf.keras.layers.Dense(10, activation="softmax")]
    return tf.keras.Sequential(layers)


def save_charts(histories, confusion, output, plt, np):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        epochs = np.arange(1, len(history["accuracy"]) + 1)
        axes[0].plot(epochs, history["accuracy"], "--", label=f"{name} train")
        axes[0].plot(epochs, history["val_accuracy"], label=f"{name} validation")
        axes[1].plot(epochs, history["loss"], "--", label=f"{name} train")
        axes[1].plot(epochs, history["val_loss"], label=f"{name} validation")
    for axis, label in zip(axes, ("Accuracy", "Loss")):
        axis.set(xlabel="Epoch", ylabel=label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    axes[1].set_title("Loss differs between shallow MLP (MSE) and deep MLP (CE)")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)
    for name, matrix in confusion.items():
        fig, axis = plt.subplots(figsize=(6, 5))
        plot = axis.imshow(matrix, cmap="Blues")
        axis.set(xlabel="Predicted digit", ylabel="True digit", title=f"{name}: official MNIST test set")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        for row in range(10):
            for col in range(10):
                axis.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=7,
                          color="white" if matrix[row, col] > matrix.max() / 2 else "black")
        fig.colorbar(plot, ax=axis)
        fig.tight_layout()
        fig.savefig(output / f"{name}_confusion_matrix.png", dpi=160)
        plt.close(fig)


def main():
    args = parse_args()
    # Imports are deferred so --help works without the training dependencies.
    import numpy as np
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    (raw_x, raw_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_idx, val_idx = split_indices(raw_y, args.validation_fraction, args.seed, np)
    if args.limit_train is not None:
        train_idx = train_idx[:args.limit_train]
    if args.limit_validation is not None:
        val_idx = val_idx[:args.limit_validation]
    train_x = raw_x[train_idx].reshape(-1, 784).astype("float32") / 255.0
    val_x = raw_x[val_idx].reshape(-1, 784).astype("float32") / 255.0
    test_x = test_x.reshape(-1, 784).astype("float32") / 255.0
    train_y = tf.keras.utils.to_categorical(raw_y[train_idx], 10)
    val_y = tf.keras.utils.to_categorical(raw_y[val_idx], 10)
    test_targets = tf.keras.utils.to_categorical(test_y, 10)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "split_indices.npz", train=train_idx, validation=val_idx)
    experiments = []
    if args.model in ("all", "comparison"):
        experiments.extend(("shallow_sgd", "shallow_adam"))
    if args.model in ("all", "deep"):
        experiments.append("deep_mlp")
    # The two optimizers start from exactly the same weights.
    tf.keras.utils.set_random_seed(args.seed)
    initial_weights = build_model(tf).get_weights()
    histories, matrices, metrics = {}, {}, {}
    for name in experiments:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.seed)
        deep = name == "deep_mlp"
        model = build_model(tf, deep=deep)
        if not deep:
            model.set_weights(initial_weights)
        if name == "shallow_sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        elif name == "shallow_adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        loss = "categorical_crossentropy" if deep else "mean_squared_error"
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        model_file = args.output / f"{name}.keras"
        history = model.fit(
            train_x, train_y, validation_data=(val_x, val_y),
            epochs=args.epochs, batch_size=args.batch_size, shuffle=True, verbose=2,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(
                str(model_file), monitor="val_loss", save_best_only=True)])
        histories[name] = {key: [float(value) for value in values]
                           for key, values in history.history.items()}
        # Only the validation set selects the saved checkpoint; test is used afterwards.
        best_model = tf.keras.models.load_model(model_file)
        test_metrics = best_model.evaluate(test_x, test_targets, verbose=0, return_dict=True)
        predicted = np.argmax(best_model.predict(test_x, verbose=0), axis=1)
        matrix = np.zeros((10, 10), dtype=np.int64)
        np.add.at(matrix, (test_y, predicted), 1)
        matrices[name] = matrix
        metrics[name] = {
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "best_validation_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            "loss_function": loss,
            "model": model_file.name,
            "confusion_matrix": matrix.tolist(),
        }
        print(f"{name}: test accuracy = {metrics[name]['test_accuracy']:.4f}")
    settings = vars(args).copy()
    settings["output"] = str(args.output)
    summary = {"settings": settings,
               "counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_y)},
               "checkpoint_selection": "minimum validation loss; test set is not used for selection",
               "tensorflow_version": tf.__version__, "metrics": metrics}
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    save_charts(histories, matrices, args.output, plt, np)
    print(f"Saved models, metrics and charts to {args.output.resolve()}")


if __name__ == "__main__":
    main()
