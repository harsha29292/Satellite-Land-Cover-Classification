"""Training entry point with early stopping and checkpointing."""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    from .config import CONFIG
    from .dataset import get_dataloaders
    from .evaluate import (
        build_classification_report,
        build_confusion_matrix,
        collect_predictions,
        confidence_analysis,
        per_class_error_rate,
    )
    from .model import ResNetWithDropout, freeze_all, unfreeze_all, unfreeze_layer4_fc
    from .uncertainty import run_mc_dropout_on_loader, system_performance, threshold_analysis
    from .utils import plot_confusion_matrix, plot_loss_curves
except ImportError:
    from config import CONFIG
    from dataset import get_dataloaders
    from evaluate import (
        build_classification_report,
        build_confusion_matrix,
        collect_predictions,
        confidence_analysis,
        per_class_error_rate,
    )
    from model import ResNetWithDropout, freeze_all, unfreeze_all, unfreeze_layer4_fc
    from uncertainty import run_mc_dropout_on_loader, system_performance, threshold_analysis
    from utils import plot_confusion_matrix, plot_loss_curves


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch and return average loss and accuracy."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    """Validate model for one epoch and return average loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train_with_checkpointing(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    save_path="best_model.pth",
):
    """Train with best-checkpoint saving based on validation accuracy."""
    best_val_acc = 0.0
    best_epoch = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                save_path,
            )
            print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f} <- saved best")
        else:
            print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f}")

    print(f"\nBest model: epoch {best_epoch} | Val Acc: {best_val_acc:.4f}")
    return history


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    patience=3,
    save_path="best_model.pth",
):
    """Train with early stopping and checkpointing."""
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                save_path,
            )
            print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f} <- saved best")
        else:
            epochs_no_improve += 1
            print(
                f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.4f} | "
                f"No improve: {epochs_no_improve}/{patience}"
            )

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest model: epoch {best_epoch} | Val Acc: {best_val_acc:.4f}")
    return history


def build_model_for_training(device):
    """Construct model and apply freezing strategy from config."""
    model = ResNetWithDropout(dropout_rate=CONFIG["model"]["dropout_rate"])

    mode = CONFIG["training"]["unfreeze_mode"]
    freeze_all(model)
    if mode == "layer4_fc":
        unfreeze_layer4_fc(model)
    elif mode == "all":
        unfreeze_all(model)

    return model.to(device)


def run_full_pipeline():
    """Run training, evaluation, and uncertainty analysis end-to-end."""
    set_seed(CONFIG["seed"])
    device = CONFIG["device"]

    train_loader, val_loader = get_dataloaders()
    model = build_model_for_training(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["training"]["lr"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )

    history = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=CONFIG["training"]["epochs"],
        device=device,
        patience=CONFIG["training"]["patience"],
        save_path=CONFIG["training"]["checkpoint_path"],
    )

    checkpoint = torch.load(CONFIG["training"]["checkpoint_path"])
    model.load_state_dict(checkpoint["model_state_dict"])

    all_preds, all_labels, all_probs = collect_predictions(model, val_loader, device)
    print(build_classification_report(all_labels, all_preds))

    cm = build_confusion_matrix(all_labels, all_preds)
    per_class_stats = per_class_error_rate(all_labels, all_preds)
    conf_stats = confidence_analysis(all_labels, all_preds, all_probs)

    print("\n=== Per Class Error Rate ===")
    for row in per_class_stats:
        print(
            f"{row['class_name']:>20} | Errors: {row['errors']:>3} / "
            f"{row['total']:>4} | Error rate: {row['error_rate']:.4f}"
        )

    print("\n=== Confidence Analysis ===")
    print(f"Correct predictions - avg confidence: {conf_stats['correct_avg_conf']:.1f}%")
    print(f"Wrong predictions   - avg confidence: {conf_stats['wrong_avg_conf']:.1f}%")
    print(
        f"Wrong predictions   - confident (>80%): "
        f"{conf_stats['wrong_confident_over_80']}"
    )
    print(
        f"Wrong predictions   - uncertain (<50%): "
        f"{conf_stats['wrong_uncertain_under_50']}"
    )

    _, all_pred_class, all_uncertainty, all_labels_mc = run_mc_dropout_on_loader(
        model, val_loader, device
    )
    wrong_mask = all_pred_class != all_labels_mc
    results_thresh, best = threshold_analysis(all_uncertainty, wrong_mask)

    if best is not None:
        perf = system_performance(
            all_pred_class=all_pred_class,
            all_labels_mc=all_labels_mc,
            all_uncertainty=all_uncertainty,
            threshold=best["threshold"],
        )
        print("\n=== Uncertainty Threshold ===")
        print(f"Best threshold:  {best['threshold']:.6f}")
        print(f"Precision:       {best['precision']:.4f}")
        print(f"Recall:          {best['recall']:.4f}")
        print(f"Flagged:         {best['flagged']} predictions")

        print("\n=== System Performance ===")
        print(f"Unflagged predictions: {perf['unflagged_count']}")
        print(f"Unflagged accuracy:    {perf['unflagged_acc']:.4f}")
        print(f"Flagged for review:    {perf['flagged_count']}")
        print(f"Flagged accuracy:      {perf['flagged_acc']:.4f}")
        print(f"Overall accuracy:      {perf['overall_acc']:.4f}")

    plot_loss_curves(history)
    plot_confusion_matrix(cm, CONFIG["class_names"], title="Confusion Matrix - ResNet18 Fine-tuned")

    return {
        "history": history,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
        "confusion_matrix": cm,
        "threshold_results": results_thresh,
    }


if __name__ == "__main__":
    run_full_pipeline()
