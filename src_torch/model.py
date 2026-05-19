from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


BASE_FILTERS = (32, 64, 128, 256, 512)
ARCHITECTURES = (
    "light_unet",
    "residual_unet",
    "attention_unet",
    "unetpp",
    "inception_unet",
    "se_unet",
    "dense_unet",
)


def _conv3(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


def _conv1(in_channels: int, out_channels: int, bias: bool = False) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)


def _activation() -> nn.LeakyReLU:
    return nn.LeakyReLU(negative_slope=0.1, inplace=True)


def scaled_filters(width_multiplier: float, base_filters: tuple[int, int, int, int, int] = BASE_FILTERS) -> tuple[int, int, int, int, int]:
    filters = []
    for value in base_filters:
        scaled = max(8, int(round(value * width_multiplier / 8.0)) * 8)
        filters.append(scaled)
    return tuple(filters)  # type: ignore[return-value]


def normalize_architecture(name: str) -> str:
    key = name.lower().replace("-", "_").replace("+", "p")
    aliases = {
        "paper": "light_unet",
        "unet": "light_unet",
        "light": "light_unet",
        "lightweight_unet": "light_unet",
        "resunet": "residual_unet",
        "res_unet": "residual_unet",
        "attention": "attention_unet",
        "attention_unet": "attention_unet",
        "att_unet": "attention_unet",
        "unetpp": "unetpp",
        "unetplusplus": "unetpp",
        "unet_p_p": "unetpp",
        "inception": "inception_unet",
        "inception_unet": "inception_unet",
        "se": "se_unet",
        "squeeze_excitation_unet": "se_unet",
        "se_unet": "se_unet",
        "dense": "dense_unet",
        "denseunet": "dense_unet",
        "dense_unet": "dense_unet",
    }
    normalized = aliases.get(key, key)
    if normalized not in ARCHITECTURES:
        valid = ", ".join(ARCHITECTURES)
        raise ValueError(f"Unknown architecture '{name}'. Valid options: {valid}.")
    return normalized


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _conv3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            _activation(),
            nn.Dropout2d(dropout),
            _conv3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            _activation(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.main = nn.Sequential(
            _conv3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            _activation(),
            nn.Dropout2d(dropout),
            _conv3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(_conv1(in_channels, out_channels), nn.BatchNorm2d(out_channels))
        self.out = nn.Sequential(_activation(), nn.Dropout2d(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.main(x) + self.shortcut(x))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _conv1(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            _conv1(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class SEConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, dropout)
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        branch_channels = _split_channels(out_channels, 4)
        b1, b2, b3, b4 = branch_channels
        self.branch1 = nn.Sequential(_conv1(in_channels, b1), nn.BatchNorm2d(b1), _activation())
        self.branch2 = nn.Sequential(_conv1(in_channels, b2), nn.BatchNorm2d(b2), _activation(), _conv3(b2, b2), nn.BatchNorm2d(b2), _activation())
        self.branch3 = nn.Sequential(_conv1(in_channels, b3), nn.BatchNorm2d(b3), _activation(), _conv3(b3, b3), nn.BatchNorm2d(b3), _activation(), _conv3(b3, b3), nn.BatchNorm2d(b3), _activation())
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), _conv1(in_channels, b4), nn.BatchNorm2d(b4), _activation())
        self.out = nn.Sequential(_conv1(out_channels, out_channels), nn.BatchNorm2d(out_channels), _activation(), nn.Dropout2d(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.out(y)


class DenseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2, layers: int = 4) -> None:
        super().__init__()
        growth = max(8, out_channels // 4)
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(channels),
                    _activation(),
                    _conv3(channels, growth),
                    nn.Dropout2d(dropout),
                )
            )
            channels += growth
        self.transition = nn.Sequential(_conv1(channels, out_channels), nn.BatchNorm2d(out_channels), _activation(), nn.Dropout2d(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.transition(torch.cat(features, dim=1))


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        inter_channels = max(8, inter_channels)
        self.gate_proj = nn.Sequential(_conv1(gate_channels, inter_channels), nn.BatchNorm2d(inter_channels))
        self.skip_proj = nn.Sequential(_conv1(skip_channels, inter_channels), nn.BatchNorm2d(inter_channels))
        self.psi = nn.Sequential(_activation(), _conv1(inter_channels, 1, bias=True), nn.Sigmoid())

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        weights = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * weights


def _split_channels(total: int, n: int) -> list[int]:
    base = total // n
    channels = [base] * n
    for idx in range(total - base * n):
        channels[idx] += 1
    return channels


def _upsample(x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
    if skip is None:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)


class UNet(nn.Module):
    """Lightweight U-Net matching the TensorFlow model topology used by the paper."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        filters: tuple[int, int, int, int, int] = BASE_FILTERS,
        dropout: float = 0.2,
        block_factory: Callable[[int, int, float], nn.Module] = ConvBlock,
    ) -> None:
        super().__init__()
        f1, f2, f3, f4, f5 = filters
        self.enc1 = block_factory(in_channels, f1, dropout)
        self.enc2 = block_factory(f1, f2, dropout)
        self.enc3 = block_factory(f2, f3, dropout)
        self.enc4 = block_factory(f3, f4, dropout)
        self.bridge = block_factory(f4, f5, dropout)

        self.dec4 = block_factory(f5 + f4, f4, dropout)
        self.dec3 = block_factory(f4 + f3, f3, dropout)
        self.dec2 = block_factory(f3 + f2, f2, dropout)
        self.dec1 = block_factory(f2 + f1, f1, dropout)
        self.out = nn.Conv2d(f1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2))
        b = self.bridge(F.max_pool2d(x4, 2))

        y = _upsample(b, x4)
        y = self.dec4(torch.cat([y, x4], dim=1))
        y = _upsample(y, x3)
        y = self.dec3(torch.cat([y, x3], dim=1))
        y = _upsample(y, x2)
        y = self.dec2(torch.cat([y, x2], dim=1))
        y = _upsample(y, x1)
        y = self.dec1(torch.cat([y, x1], dim=1))
        return self.out(y)


class AttentionUNet(UNet):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        filters: tuple[int, int, int, int, int] = BASE_FILTERS,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, filters=filters, dropout=dropout)
        f1, f2, f3, f4, f5 = filters
        self.att4 = AttentionGate(f5, f4, f4 // 2)
        self.att3 = AttentionGate(f4, f3, f3 // 2)
        self.att2 = AttentionGate(f3, f2, f2 // 2)
        self.att1 = AttentionGate(f2, f1, f1 // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x4 = self.enc4(F.max_pool2d(x3, 2))
        b = self.bridge(F.max_pool2d(x4, 2))

        y = _upsample(b, x4)
        y = self.dec4(torch.cat([y, self.att4(b, x4)], dim=1))
        y = _upsample(y, x3)
        y = self.dec3(torch.cat([y, self.att3(y, x3)], dim=1))
        y = _upsample(y, x2)
        y = self.dec2(torch.cat([y, self.att2(y, x2)], dim=1))
        y = _upsample(y, x1)
        y = self.dec1(torch.cat([y, self.att1(y, x1)], dim=1))
        return self.out(y)


class UNetPP(nn.Module):
    """UNet++ style dense decoder with nested skip pathways."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        filters: tuple[int, int, int, int, int] = BASE_FILTERS,
        dropout: float = 0.2,
        block_factory: Callable[[int, int, float], nn.Module] = ConvBlock,
    ) -> None:
        super().__init__()
        f1, f2, f3, f4, f5 = filters
        self.x00 = block_factory(in_channels, f1, dropout)
        self.x10 = block_factory(f1, f2, dropout)
        self.x20 = block_factory(f2, f3, dropout)
        self.x30 = block_factory(f3, f4, dropout)
        self.x40 = block_factory(f4, f5, dropout)

        self.x01 = block_factory(f1 + f2, f1, dropout)
        self.x11 = block_factory(f2 + f3, f2, dropout)
        self.x21 = block_factory(f3 + f4, f3, dropout)
        self.x31 = block_factory(f4 + f5, f4, dropout)

        self.x02 = block_factory(f1 * 2 + f2, f1, dropout)
        self.x12 = block_factory(f2 * 2 + f3, f2, dropout)
        self.x22 = block_factory(f3 * 2 + f4, f3, dropout)

        self.x03 = block_factory(f1 * 3 + f2, f1, dropout)
        self.x13 = block_factory(f2 * 3 + f3, f2, dropout)

        self.x04 = block_factory(f1 * 4 + f2, f1, dropout)
        self.out = nn.Conv2d(f1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(F.max_pool2d(x00, 2))
        x20 = self.x20(F.max_pool2d(x10, 2))
        x30 = self.x30(F.max_pool2d(x20, 2))
        x40 = self.x40(F.max_pool2d(x30, 2))

        x01 = self.x01(torch.cat([x00, _upsample(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, _upsample(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, _upsample(x30, x20)], dim=1))
        x31 = self.x31(torch.cat([x30, _upsample(x40, x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, _upsample(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, _upsample(x21, x10)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, _upsample(x31, x20)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, _upsample(x12, x00)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, _upsample(x22, x10)], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, _upsample(x13, x00)], dim=1))
        return self.out(x04)


def build_model(
    architecture: str = "light_unet",
    in_channels: int = 1,
    num_classes: int = 6,
    filters: tuple[int, int, int, int, int] = BASE_FILTERS,
    dropout: float = 0.2,
) -> nn.Module:
    architecture = normalize_architecture(architecture)
    kwargs = {"in_channels": in_channels, "num_classes": num_classes, "filters": filters, "dropout": dropout}
    if architecture == "light_unet":
        return UNet(**kwargs)
    if architecture == "residual_unet":
        return UNet(**kwargs, block_factory=ResidualConvBlock)
    if architecture == "attention_unet":
        return AttentionUNet(**kwargs)
    if architecture == "unetpp":
        return UNetPP(**kwargs)
    if architecture == "inception_unet":
        return UNet(**kwargs, block_factory=InceptionBlock)
    if architecture == "se_unet":
        return UNet(**kwargs, block_factory=SEConvBlock)
    if architecture == "dense_unet":
        return UNet(**kwargs, block_factory=DenseConvBlock)
    raise AssertionError(f"Unhandled architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_conv_macs(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 1, 256, 256),
    device: torch.device | str = "cpu",
) -> int:
    """Estimate Conv2d MACs with a real forward pass.

    The estimate counts convolution multiply-accumulate operations only. Batch
    norm, activation, pooling, concatenation, interpolation, and memory movement
    are intentionally excluded, so measured latency is still needed for hardware
    conclusions.
    """
    model = model.to(device)
    model_was_training = model.training
    model.eval()
    handles = []
    macs = 0

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if not isinstance(module, nn.Conv2d):
            return
        batch, cout, height, width = output.shape
        cin = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups
        macs += int(batch * cout * height * width * (cin // groups) * kh * kw)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        dummy = torch.zeros(input_shape, device=device)
        model(dummy)

    for handle in handles:
        handle.remove()
    model.train(model_was_training)
    return macs
