"""Evaluation utilities: reports, confusion analysis, and confidence analysis."""

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


def collect_predictions(model, loader, device):
    """Collect predictions, labels, and class probabilities from a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def build_classification_report(all_labels, all_preds):
    """Generate sklearn classification report text."""
    return classification_report(all_labels, all_preds, target_names=CONFIG["class_names"])


def build_confusion_matrix(all_labels, all_preds):
    """Generate confusion matrix array."""
    return confusion_matrix(all_labels, all_preds)


def per_class_error_rate(all_labels, all_preds):
    """Compute per-class error statistics."""
    class_names = CONFIG["class_names"]
    results = []
    for i, cls in enumerate(class_names):
        cls_idx = np.where(all_labels == i)[0]
        cls_wrong = np.where((all_labels == i) & (all_preds != i))[0]
        error_rate = len(cls_wrong) / len(cls_idx)
        results.append(
            {
                "class_name": cls,
                "errors": len(cls_wrong),
                "total": len(cls_idx),
                "error_rate": error_rate,
            }
        )
    return results


def confidence_analysis(all_labels, all_preds, all_probs):
    """Compute confidence distribution stats for correct and wrong predictions."""
    wrong_confidences = []
    correct_confidences = []

    for idx in range(len(all_labels)):
        conf = all_probs[idx][all_preds[idx]] * 100
        if all_preds[idx] == all_labels[idx]:
            correct_confidences.append(conf)
        else:
            wrong_confidences.append(conf)

    return {
        "correct_avg_conf": float(np.mean(correct_confidences)),
        "wrong_avg_conf": float(np.mean(wrong_confidences)),
        "wrong_confident_over_80": int(sum(1 for c in wrong_confidences if c > 80)),
        "wrong_uncertain_under_50": int(sum(1 for c in wrong_confidences if c < 50)),
        "correct_confidences": correct_confidences,
        "wrong_confidences": wrong_confidences,
    }


def top_confusion_pairs(cm):
    """Extract off-diagonal confusion pairs sorted by frequency."""
    confusion_pairs = []
    class_names = CONFIG["class_names"]

    for true_cls in range(CONFIG["num_classes"]):
        for pred_cls in range(CONFIG["num_classes"]):
            if true_cls != pred_cls and cm[true_cls][pred_cls] > 0:
                confusion_pairs.append(
                    {
                        "true": class_names[true_cls],
                        "pred": class_names[pred_cls],
                        "count": int(cm[true_cls][pred_cls]),
                    }
                )

    confusion_pairs = sorted(confusion_pairs, key=lambda x: x["count"], reverse=True)
    return confusion_pairs
