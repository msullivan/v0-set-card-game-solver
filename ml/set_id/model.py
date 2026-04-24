"""Two backbones, both sharing a 4-head interface.

Forward returns logits in the order (number, color, shape, shading).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class _Heads(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.number = nn.Linear(in_dim, 3)
        self.color = nn.Linear(in_dim, 3)
        self.shape = nn.Linear(in_dim, 3)
        self.shading = nn.Linear(in_dim, 3)

    def forward(self, f: torch.Tensor):
        return self.number(f), self.color(f), self.shape(f), self.shading(f)


class Freezable(nn.Module):
    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True


class ResNet18Net(Freezable):
    feature_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.heads = _Heads(self.feature_dim)

    def forward(self, x: torch.Tensor):
        f = self.features(x).flatten(1)
        return self.heads(f)


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallNet(Freezable):
    """~1.5M??-param custom CNN intended for browser deployment."""

    feature_dim = 256

    def __init__(self, pretrained: bool = False):
        super().__init__()
        del pretrained  # no pretrained weights for custom backbone
        self.features = nn.Sequential(
            _ConvBlock(3, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.heads = _Heads(self.feature_dim)

    def forward(self, x: torch.Tensor):
        f = self.features(x).flatten(1)
        return self.heads(f)


class SmallerNet(Freezable):
    """???-param custom CNN intended for browser deployment."""

    feature_dim = 128

    def __init__(self, pretrained: bool = False):
        super().__init__()
        del pretrained  # no pretrained weights for custom backbone
        self.features = nn.Sequential(
            _ConvBlock(3, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.heads = _Heads(self.feature_dim)

    def forward(self, x: torch.Tensor):
        f = self.features(x).flatten(1)
        return self.heads(f)


ARCHES = {"resnet18": ResNet18Net, "small": SmallNet, "smaller": SmallerNet}


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHES)}")
    return ARCHES[arch](pretrained=pretrained)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
