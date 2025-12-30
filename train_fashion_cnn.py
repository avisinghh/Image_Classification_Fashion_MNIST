#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a stronger CNN on Fashion-MNIST and save:
- best model (val_accuracy)      -> fashion_cnn.keras (default)
- training curves                -> training_curves.png
- confusion matrix               -> confusion_matrix.png
- sample predictions             -> sample_predictions.png
- classification report (txt)    -> classification_report.txt
- per-epoch CSV log              -> training_log.csv

Run examples:
  python train_fashion_cnn.py
  python train_fashion_cnn.py --epochs 12 --batch 128 --lr 0.001
  python train_fashion_cnn.py --label-smoothing 0.05
  python train_fashion_cnn.py --mixed auto      # CUDA only; auto-disables on METAL (Mac)
"""

import os, argparse, numpy as np, tensorflow as tf, seaborn as sns
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# -------- Matplotlib (GUI if possible, else headless) --------
import matplotlib
try: matplotlib.use("TkAgg")
except Exception: matplotlib.use("Agg")
import matplotlib.pyplot as plt
sns.set_context("notebook")

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

# ---------------- Utilities ----------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def pick_adamw(lr: float, wd: float):
    # Try stable AdamW, else experimental, else Adam
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    except Exception:
        try:
            return tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=wd)
        except Exception:
            return tf.keras.optimizers.Adam(learning_rate=lr)

def enable_mixed_precision(mode: str) -> bool:
    """
    mode: 'auto' -> enable only on CUDA (not on METAL), 'on' -> force, 'off' -> disable
    Returns True if enabled.
    """
    if mode == "off":
        print("ℹ️ Mixed precision disabled.")
        return False

    gpus = tf.config.list_logical_devices("GPU")
    if not gpus:
        print("ℹ️ No GPU found; mixed precision disabled.")
        return False

    # Detect Apple METAL and avoid enabling there for stability/perf on TF 2.12
    is_metal = any("METAL" in d.name.upper() for d in gpus)
    if mode == "auto" and is_metal:
        print("ℹ️ Apple METAL GPU detected — keeping float32 (use --mixed on to force).")
        return False

    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("✅ Mixed precision enabled (policy='mixed_float16').")
        return True
    except Exception as e:
        print("⚠️ Could not enable mixed precision:", e)
        return False

def build_model(input_shape=(28, 28, 1), lr=1e-3, wd=1e-5, aug=True, loss_obj=None):
    """
    Stronger CNN:
    - Keras preprocessing augmentation on-GPU
    - Conv blocks with BatchNorm + ReLU + Dropout
    - GAP + Dense head (Dropout)
    - Outputs forced to float32 for numerical stability with mixed precision
    """
    inputs = layers.Input(shape=input_shape)

    x = inputs
    if aug:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.05)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomTranslation(0.1, 0.1)(x)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
        return x

    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(10, activation="softmax", dtype="float32")(x)
    model = models.Model(inputs, outputs)

    opt = pick_adamw(lr, wd)
    model.compile(optimizer=opt, loss=loss_obj, metrics=["accuracy"])
    return model

def save_and_maybe_show(fig: plt.Figure, path: str, show: bool):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        try: plt.show()
        except Exception: pass
    plt.close(fig)

def plot_training_curves(history, out_path: str, show: bool):
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(h["loss"], label="train"); ax1.plot(h["val_loss"], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(h["accuracy"], label="train"); ax2.plot(h["val_accuracy"], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    fig.suptitle("Training Curves"); fig.tight_layout()
    save_and_maybe_show(fig, out_path, show)

def plot_confusion_matrix(y_true, y_pred, out_path: str, show: bool):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, rotation=0, fontsize=8)
    fig.tight_layout()
    save_and_maybe_show(fig, out_path, show)

def plot_sample_preds(x, y_true, y_pred, k: int, out_path: str, show: bool):
    k = max(1, min(k, len(x)))
    idxs = np.random.choice(len(x), size=k, replace=False)
    fig, axes = plt.subplots(1, k, figsize=(k*1.6, 2.2))
    if k == 1: axes = [axes]
    for ax, i in zip(axes, idxs):
        ax.imshow(x[i].squeeze(), cmap="gray")
        t = CLASS_NAMES[int(y_true[i])]; p = CLASS_NAMES[int(y_pred[i])]
        color = "green" if y_true[i] == y_pred[i] else "red"
        ax.set_title(f"P: {p}\nT: {t}", fontsize=9, color=color)
        ax.axis("off")
    fig.suptitle("Actual vs Predicted", y=1.05)
    fig.tight_layout()
    save_and_maybe_show(fig, out_path, show)

def make_datasets(x_trn, y_trn, x_val, y_val, batch, one_hot: bool):
    AUTOTUNE = tf.data.AUTOTUNE

    def _map(x, y):
        if one_hot:
            y = tf.one_hot(y, depth=10, dtype=tf.float32)
        return x, y

    ds_trn = (tf.data.Dataset.from_tensor_slices((x_trn, y_trn))
              .shuffle(len(x_trn), reshuffle_each_iteration=True)
              .map(_map, num_parallel_calls=AUTOTUNE)
              .batch(batch).cache().prefetch(AUTOTUNE))

    ds_val = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
              .map(_map, num_parallel_calls=AUTOTUNE)
              .batch(batch).cache().prefetch(AUTOTUNE))
    return ds_trn, ds_val

def main(args):
    # Paths (save next to script)
    out_dir = script_dir()
    model_path = os.path.join(out_dir, args.out)
    curves_png = os.path.join(out_dir, "training_curves.png")
    cm_png     = os.path.join(out_dir, "confusion_matrix.png")
    samples_png= os.path.join(out_dir, "sample_predictions.png")
    report_txt = os.path.join(out_dir, "classification_report.txt")
    log_csv    = os.path.join(out_dir, "training_log.csv")

    print("Saving to:")
    print("  model:", model_path)
    print("  curves:", curves_png)
    print("  cm:", cm_png)
    print("  samples:", samples_png)
    print("  report:", report_txt)
    print("  log:", log_csv)
    print()

    # Seed & mixed precision
    set_seed(args.seed)
    enabled_mp = enable_mixed_precision(args.mixed)

    # GPU info
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        except Exception: pass
    print(f"TensorFlow {tf.__version__} | GPUs detected: {len(gpus)}\n")

    # Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype("float32")/255.0)[..., None]
    x_test  = (x_test.astype("float32") /255.0)[..., None]
    print("Raw sizes -> Train:", x_train.shape, " | Test:", x_test.shape)

    # Stratified split for validation
    x_trn, x_val, y_trn, y_val = train_test_split(
        x_train, y_train, test_size=args.val_split, random_state=args.seed, stratify=y_train
    )
    print(f"Split -> x_trn: {x_trn.shape} | x_val: {x_val.shape} | x_test: {x_test.shape}")

    # Choose loss + label format
    use_label_smoothing = args.label_smoothing > 0.0
    if use_label_smoothing:
        loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        one_hot = True
        print(f"Using CategoricalCrossentropy with label_smoothing={args.label_smoothing}")
    else:
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        one_hot = False
        print("Using SparseCategoricalCrossentropy (no label smoothing)")

    # Datasets
    ds_trn, ds_val = make_datasets(x_trn, y_trn, x_val, y_val, args.batch, one_hot=one_hot)

    # Model
    model = build_model(input_shape=(28,28,1), lr=args.lr, wd=args.weight_decay,
                        aug=not args.no_aug, loss_obj=loss_obj)
    model.summary()

    # Callbacks (EarlyStopping removed; ReduceLROnPlateau patience set to a valid int)
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                           save_best_only=True, save_weights_only=False, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.CSVLogger(log_csv, append=False)
    ]

    # Train
    history = model.fit(
        ds_trn,
        validation_data=ds_val,
        epochs=args.epochs,
        verbose=2,
        callbacks=cbs
    )

    # Training curves
    plot_training_curves(history, curves_png, show=not args.no_show)

    # Evaluate on test (match label format!)
    if one_hot:
        y_test_eval = tf.one_hot(y_test, depth=10, dtype=tf.float32)
    else:
        y_test_eval = y_test
    test_loss, test_acc = model.evaluate(x_test, y_test_eval, verbose=0)
    print(f"\n✅ Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Predictions for reports (keep y_test as integers)
    y_probs = model.predict(x_test, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)

    # Confusion matrix + report
    plot_confusion_matrix(y_test, y_pred, cm_png, show=not args.no_show)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\nClassification report:\n", report)
    with open(report_txt, "w") as f: f.write(report)

    # Sample predictions grid
    plot_sample_preds(x_test, y_test, y_pred, k=args.samples, out_path=samples_png, show=not args.no_show)

    print(f"\n💾 Best model saved to: {model_path}")
    print("🖼️ Plots saved:")
    print("   -", curves_png)
    print("   -", cm_png)
    print("   -", samples_png)
    print("📄 Report saved:", report_txt)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--val-split", type=float, default=0.1, dest="val_split")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help=">0 uses CategoricalCrossentropy with one-hot labels (compatible with TF 2.12).")
    p.add_argument("--mixed", type=str, choices=["auto","on","off"], default="auto",
                   help="Enable mixed precision (auto: CUDA only; disabled on METAL).")
    p.add_argument("--no-aug", action="store_true", help="Disable data augmentation layers.")
    p.add_argument("--out", type=str, default="fashion_cnn.keras")
    p.add_argument("--samples", type=int, default=10, help="How many sample predictions to show")
    p.add_argument("--no-show", action="store_true", help="Save plots only; do not open windows")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
