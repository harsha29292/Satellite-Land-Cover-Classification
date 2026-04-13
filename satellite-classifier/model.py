"""Model definitions and parameter freezing helpers."""

import torch.nn as nn
import torchvision.models as models

try:
    from .config import CONFIG
except ImportError:
    from config import CONFIG


class ResNetWithDropout(nn.Module):
    """ResNet18 model with Dropout before the final classifier."""

    def __init__(self, dropout_rate=0.5):
        """Initialize the dropout-augmented ResNet18 model."""
        super().__init__()
        model_cfg = CONFIG["model"]
        self.base = models.resnet18(weights=model_cfg["weights"])
        self.base.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, CONFIG["num_classes"]),
        )

    def forward(self, x):
        """Run a forward pass through the network."""
        return self.base(x)


def build_resnet18():
    """Build the baseline ResNet18 classifier without dropout wrapper."""
    model_cfg = CONFIG["model"]
    model = models.resnet18(weights=model_cfg["weights"])
    model.fc = nn.Linear(512, CONFIG["num_classes"])
    return model


def freeze_all(model):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_layer4_fc(model):
    """Unfreeze layer4 and final classification head parameters."""
    if hasattr(model, "layer4") and hasattr(model, "fc"):
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        return

    if hasattr(model, "base") and hasattr(model.base, "layer4") and hasattr(model.base, "fc"):
        for param in model.base.layer4.parameters():
            param.requires_grad = True
        for param in model.base.fc.parameters():
            param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
