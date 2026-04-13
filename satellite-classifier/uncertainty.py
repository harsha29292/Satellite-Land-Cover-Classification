"""Monte Carlo Dropout uncertainty estimation helpers."""

import numpy as np
import torch
from tqdm import tqdm

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


def enable_dropout(model):
    """Keep dropout layers active during inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(model, images, n_passes=20):
    """Run multiple stochastic forward passes and return mean prediction and uncertainty."""
    model.eval()
    enable_dropout(model)

    predictions = []

    with torch.no_grad():
        for _ in range(n_passes):
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predictions.append(probs.unsqueeze(0))

    predictions = torch.cat(predictions, dim=0)
    mean_probs = predictions.mean(dim=0)
    uncertainty = predictions.var(dim=0).mean(dim=1)
    pred_class = mean_probs.argmax(dim=1)

    return mean_probs, pred_class, uncertainty


def run_mc_dropout_on_loader(model, loader, device):
    """Run MC Dropout over a dataloader and aggregate outputs."""
    all_mean_probs = []
    all_pred_class = []
    all_uncertainty = []
    all_labels = []

    model.eval()
    for images, labels in tqdm(loader, desc="MC Dropout"):
        images = images.to(device)
        mean_probs, pred_class, uncertainty = mc_dropout_predict(
            model, images, n_passes=CONFIG["uncertainty"]["mc_passes"]
        )
        all_mean_probs.extend(mean_probs.cpu().numpy())
        all_pred_class.extend(pred_class.cpu().numpy())
        all_uncertainty.extend(uncertainty.cpu().numpy())
        all_labels.extend(labels.numpy())

    return (
        np.array(all_mean_probs),
        np.array(all_pred_class),
        np.array(all_uncertainty),
        np.array(all_labels),
    )


def threshold_analysis(all_uncertainty, wrong_mask):
    """Evaluate precision and recall across uncertainty thresholds."""
    ucfg = CONFIG["uncertainty"]
    thresholds = np.linspace(
        ucfg["threshold_min"], ucfg["threshold_max"], ucfg["threshold_steps"]
    )

    results_thresh = []
    for thresh in thresholds:
        flagged = all_uncertainty > thresh

        if flagged.sum() > 0:
            precision = wrong_mask[flagged].sum() / flagged.sum()
        else:
            precision = 0

        if wrong_mask.sum() > 0:
            recall = wrong_mask[flagged].sum() / wrong_mask.sum()
        else:
            recall = 0

        results_thresh.append(
            {
                "threshold": float(thresh),
                "flagged": int(flagged.sum()),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

    valid = [
        r
        for r in results_thresh
        if r["recall"] > ucfg["min_recall_for_best_threshold"]
    ]
    best = max(valid, key=lambda x: x["precision"]) if valid else None
    return results_thresh, best


def system_performance(all_pred_class, all_labels_mc, all_uncertainty, threshold):
    """Compute split performance for flagged and unflagged predictions."""
    flagged_mask = all_uncertainty > threshold
    unflagged_mask = ~flagged_mask

    unflagged_acc = (
        all_pred_class[unflagged_mask] == all_labels_mc[unflagged_mask]
    ).mean()
    flagged_acc = (all_pred_class[flagged_mask] == all_labels_mc[flagged_mask]).mean()
    overall_acc = (all_pred_class == all_labels_mc).mean()

    return {
        "flagged_count": int(flagged_mask.sum()),
        "unflagged_count": int(unflagged_mask.sum()),
        "flagged_acc": float(flagged_acc),
        "unflagged_acc": float(unflagged_acc),
        "overall_acc": float(overall_acc),
        "flagged_mask": flagged_mask,
        "unflagged_mask": unflagged_mask,
    }
