"""General utility and plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


def denormalize_image(img):
    """Denormalize a CHW image tensor array for visualization."""
    mean = np.array(CONFIG["data"]["mean"])
    std = np.array(CONFIG["data"]["std"])
    img = img.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def plot_loss_curves(history):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
