from __future__ import annotations

from typing import Any

import torch.nn as nn
import torchvision.models as models


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    # initialize final layer
    nn.init.kaiming_normal_(model.fc.weight)
    if model.fc.bias is not None:
        nn.init.zeros_(model.fc.bias)
    return model


def get_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == 'resnet50':
        return build_resnet50(num_classes, pretrained=pretrained)
    raise ValueError(f'Unknown model {name}')
