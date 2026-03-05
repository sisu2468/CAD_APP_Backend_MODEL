"""
ResNet-34 backbone U-Net for machine part segmentation.

Uses ResNet-34 encoder pretrained on ImageNet via torchvision.
Decoder is trained from scratch with skip connections.

Key advantages over custom OutlineUNet:
- Encoder already understands image features from 1M+ ImageNet images.
- Works well with very few training samples (5–50) via transfer learning.
- Higher resolution (512×512) preserves edge detail for accurate outlines.
- Differential learning rates: low LR for encoder, high LR for decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# Recommended input size — higher resolution preserves machine part edge detail
PRETRAINED_MODEL_INPUT_SIZE = 512


class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """Upsample × 2, optionally concat a skip, then double-conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _ConvBnRelu(in_ch + skip_ch, out_ch)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    """
    U-Net with ResNet-34 encoder pretrained on ImageNet.

    Encoder channel sizes (ResNet-34):
        e0: 64  @ H/2   (conv1 + bn + relu, before maxpool)
        e1: 64  @ H/4   (layer1)
        e2: 128 @ H/8   (layer2)
        e3: 256 @ H/16  (layer3)
        e4: 512 @ H/32  (layer4)

    Decoder:
        d4: 256 @ H/16  (upsample e4, skip e3)
        d3: 128 @ H/8   (upsample d4, skip e2)
        d2: 64  @ H/4   (upsample d3, skip e1)
        d1: 32  @ H/2   (upsample d2, skip e0)
        d0: 16  @ H/1   (upsample d1, no skip)
        out: 1  @ H/1
    """

    def __init__(self, out_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__()

        # ----- Encoder (ResNet-34) -----
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
        except (ImportError, TypeError, AttributeError):
            # Older torchvision (<0.13) uses pretrained=True/False
            from torchvision.models import resnet34  # type: ignore
            backbone = resnet34(pretrained=pretrained)  # type: ignore

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # H/2, 64ch
        self.pool0 = backbone.maxpool                                             # H/4
        self.enc1 = backbone.layer1                                               # H/4, 64ch
        self.enc2 = backbone.layer2                                               # H/8, 128ch
        self.enc3 = backbone.layer3                                               # H/16, 256ch
        self.enc4 = backbone.layer4                                               # H/32, 512ch

        # ----- Decoder -----
        # in_ch = channels from previous decoder; skip_ch = matching encoder skip
        self.dec4 = _DecoderBlock(512, 256, 256)   # H/16
        self.dec3 = _DecoderBlock(256, 128, 128)   # H/8
        self.dec2 = _DecoderBlock(128, 64, 64)     # H/4
        self.dec1 = _DecoderBlock(64, 64, 32)      # H/2 (skip = e0)
        self.dec0 = _DecoderBlock(32, 0, 16)       # H   (no skip)

        self.out_conv = nn.Conv2d(16, out_channels, 1)

        self._init_decoder_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)        # H/2
        ep = self.pool0(e0)      # H/4
        e1 = self.enc1(ep)       # H/4
        e2 = self.enc2(e1)       # H/8
        e3 = self.enc3(e2)       # H/16
        e4 = self.enc4(e3)       # H/32

        # Decoder with skip connections
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1)

        return self.out_conv(d0)

    # ------------------------------------------------------------------
    # Helpers for transfer learning
    # ------------------------------------------------------------------

    def _init_decoder_weights(self) -> None:
        """Kaiming init for decoder layers (encoder already has ImageNet weights)."""
        for m in [self.dec4, self.dec3, self.dec2, self.dec1, self.dec0, self.out_conv]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def freeze_encoder(self) -> None:
        """Freeze encoder — train only decoder. Use for first N epochs."""
        for module in (self.enc0, self.pool0, self.enc1, self.enc2, self.enc3, self.enc4):
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True

    def encoder_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for m in (self.enc0, self.enc1, self.enc2, self.enc3, self.enc4):
            params.extend(m.parameters())
        return params

    def decoder_parameters(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        for m in (self.dec4, self.dec3, self.dec2, self.dec1, self.dec0, self.out_conv):
            params.extend(m.parameters())
        return params


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def load_pretrained_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> ResNet34UNet:
    """Load a saved ResNet34UNet checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34UNet(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device)
    model.eval()
    return model


def is_pretrained_checkpoint(checkpoint_path: str) -> bool:
    """Return True if the checkpoint was saved from ResNet34UNet (not the old OutlineUNet)."""
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
        d = state.get("model_state_dict", state)
        return any(k.startswith("enc0.") or k.startswith("dec4.") for k in d.keys())
    except Exception:
        return False
