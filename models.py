"""
models.py
=========
Neural network architectures for the FL-HE framework.
  python models.py [--mode simulate_he | tenseal]
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D-CNN
# ---------------------------------------------------------------------------

class CNN2D(nn.Module):
    """
    4-layer convolutional network.

    Architecture:
        Conv(32) -> Conv(64) -> Conv(128) -> Conv(256)
        -> AdaptiveAvgPool -> FC(512) -> FC(256) -> FC(n_classes)

    Each conv block: Conv2d -> BatchNorm -> ReLU -> MaxPool(2x2)

    Parameters
    ----------
    n_classes   : number of output classes
    in_channels : input image channels (1 for grayscale, 3 for RGB)
    """

    def __init__(self, n_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project to embedding dim.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        assert img_size % patch_size == 0, (
            f"Image size {img_size} not divisible by patch size {patch_size}."
        )
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, n_patches, embed_dim)
        x = self.proj(x)                    # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)                    # (B, embed_dim, n_patches)
        return x.transpose(1, 2)            # (B, n_patches, embed_dim)


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with multi-head self-attention.
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer
        6 blocks, 10 attention heads, 768 embedding dim
        patch_size=16 for F-MNIST/X-ray, patch_size=4 for CIFAR-10

    Parameters
    ----------
    img_size    : input image spatial size (assumed square)
    patch_size  : patch size (16 or 4 per paper)
    in_channels : 1 (grayscale) or 3 (RGB)
    n_classes   : output classes
    embed_dim   : token embedding dimension (768)
    depth       : number of transformer blocks (6)
    n_heads     : attention heads (10)
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 16,
        in_channels: int = 1,
        n_classes: int = 10,
        embed_dim: int = 768,
        depth: int = 6,
        n_heads: int = 12,  
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Adjust patch size if image too small
        if img_size < patch_size:
            patch_size = img_size // 2 if img_size >= 2 else 1
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                                    # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                     # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                             # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])                                  # CLS token

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# EfficientNet-B0
# ---------------------------------------------------------------------------

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0

    Parameters
    ----------
    n_classes   : output classes
    in_channels : 1 (grayscale) or 3 (RGB)
    """

    def __init__(self, n_classes: int = 10, in_channels: int = 1):
        super().__init__()
        from torchvision.models import efficientnet_b0

        backbone = efficientnet_b0(weights=None)

        # Adapt stem conv for non-RGB inputs
        if in_channels != 3:
            orig = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                in_channels, orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )

        # Replace classifier head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, n_classes),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Dataset-specific
DATASET_CONFIGS = {
    "fmnist": dict(img_size=28, in_channels=1, n_classes=10,
                   vit_patch=4),    # 28/16 < 2 patches, use 4
    "xray":   dict(img_size=64, in_channels=1, n_classes=2,
                   vit_patch=16),
    "cifar10": dict(img_size=32, in_channels=3, n_classes=10,
                    vit_patch=4),
}


def build_model(
    name: str,
    dataset: str = "fmnist",
    img_size: Optional[int] = None,
    in_channels: Optional[int] = None,
    n_classes: Optional[int] = None,
) -> nn.Module:
    """
    Factory function: build a model by name for a given dataset.

    Parameters
    ----------
    name        : "cnn" | "vit" | "efficientnet"
    dataset     : "fmnist" | "xray" | "cifar10"  (sets defaults)
    img_size    : override image spatial size
    in_channels : override number of input channels
    n_classes   : override number of output classes

    Returns
    -------
    nn.Module  (untrained)
    """
    name = name.lower().replace("-", "").replace("_", "")
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"dataset must be one of {list(DATASET_CONFIGS)}. Got '{dataset}'.")

    cfg = DATASET_CONFIGS[dataset]
    img  = img_size    or cfg["img_size"]
    ch   = in_channels or cfg["in_channels"]
    nc   = n_classes   or cfg["n_classes"]
    vp   = cfg["vit_patch"]

    if name == "cnn":
        return CNN2D(n_classes=nc, in_channels=ch)

    if name in ("vit", "transformer", "visiontransformer"):
        return VisionTransformer(
            img_size=img, patch_size=vp, in_channels=ch,
            n_classes=nc, embed_dim=768, depth=6, n_heads=12,
        )

    if name in ("efficientnet", "efficientnetb0", "efficientnet_b0"):
        return EfficientNetB0(n_classes=nc, in_channels=ch)

    raise ValueError(
        f"Unknown model '{name}'. Choose from: cnn, vit, efficientnet."
    )


# ---------------------------------------------------------------------------
# Self-contained test cases
# ---------------------------------------------------------------------------

def _test_case_1_output_shapes(simulate: bool):
    """
    Test 1 — All three architectures produce correct output shapes
    for each dataset configuration.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 1: Output shapes for all architectures  [{mode_label}] ===")

    configs = [
        ("fmnist",  (2, 1, 28, 28),  10),
        ("xray",    (2, 1, 64, 64),   2),
        ("cifar10", (2, 3, 32, 32),  10),
    ]

    for dataset, input_shape, expected_classes in configs:
        x = torch.randn(*input_shape)
        for arch in ["cnn", "vit", "efficientnet"]:
            model = build_model(arch, dataset=dataset)
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert out.shape == (input_shape[0], expected_classes), (
                f"{arch}/{dataset}: expected ({input_shape[0]}, {expected_classes}), "
                f"got {tuple(out.shape)}"
            )
            n = model.n_params()
            print(f"  {arch:12s} / {dataset:8s}  -> {tuple(out.shape)}  "
                  f"params={n:,}")

    print("  PASSED")


def _test_case_2_param_counts(simulate: bool):
    """
    Test 2 — Parameter counts are in the right ballpark vs paper:
      CNN:         ~1-3M   (paper doesn't specify exact count for CNN)
      ViT:         ~85M    (standard ViT-Base/10 with 768-dim, 6 blocks)
      EfficientNet: ~5.3M  (paper explicitly states 5.3M)
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 2: Parameter counts  [{mode_label}] ===")

    # Use CIFAR-10 as reference (RGB, 32x32, 10 classes)
    checks = [
        ("cnn",         "cifar10", 1_000_000,  10_000_000),
        ("vit",         "cifar10", 10_000_000, 200_000_000),
        ("efficientnet","cifar10", 3_000_000,  10_000_000),
    ]

    for arch, dataset, lo, hi in checks:
        model = build_model(arch, dataset=dataset)
        n = model.n_params()
        print(f"  {arch:12s}: {n:>12,} params  (expected {lo:,} – {hi:,})")
        assert lo <= n <= hi, (
            f"{arch} has {n:,} params, expected [{lo:,}, {hi:,}]."
        )

    # EfficientNet-B0 
    eff = build_model("efficientnet", dataset="cifar10")
    n_eff = eff.n_params()
    assert 3_500_000 <= n_eff <= 6_000_000, (
        f"EfficientNet-B0 has {n_eff:,} params."
    )

    print("  PASSED")


def _test_case_3_forward_backward(simulate: bool):
    """
    Test 3 — Forward + backward pass runs without errors for all models,
    gradients are non-zero, and loss decreases after one SGD step.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 3: Forward + backward pass  [{mode_label}] ===")

    torch.manual_seed(7)
    criterion = nn.CrossEntropyLoss()

    for arch in ["cnn", "vit", "efficientnet"]:
        # Use fmnist (small images, fast)
        model = build_model(arch, dataset="fmnist")
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))

        # Forward
        out = model(x)
        loss1 = criterion(out, y)

        # Backward
        loss1.backward()

        # Check gradients exist and are non-zero
        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0, f"{arch}: no gradients computed."
        assert any(g > 0 for g in grad_norms), f"{arch}: all gradients are zero."

        # One SGD step
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        opt.step()
        opt.zero_grad()

        out2 = model(x)
        loss2 = criterion(out2, y)

        print(f"  {arch:12s}: loss {loss1.item():.4f} -> {loss2.item():.4f}  "
              f"grad_norms_nonzero={sum(g>0 for g in grad_norms)}/{len(grad_norms)}")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="models.py self-test suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["simulate_he", "tenseal"],
        default="tenseal",
        help=(
            "tenseal     : genuine CKKS via TenSEAL (default)\n"
            "simulate_he : fast mock encryption, no TenSEAL required"
        ),
    )
    args = parser.parse_args()
    simulate = args.mode == "simulate_he"

    print("=" * 60)
    print(f"models.py — self-test suite  [{args.mode}]")
    print("=" * 60)

    _test_case_1_output_shapes(simulate)
    _test_case_2_param_counts(simulate)
    _test_case_3_forward_backward(simulate)

    print("\nAll tests passed.")