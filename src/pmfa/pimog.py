from __future__ import annotations

import sys
from pathlib import Path

import kornia
import numpy as np
import torch
from torch import Tensor, nn


def _patch_legacy_kornia_api() -> None:
    """Expose transforms where the 2022 PIMoG implementation expects them."""
    transforms = kornia.geometry.transform
    for name in (
        "get_perspective_transform",
        "warp_perspective",
        "warp_affine",
        "get_rotation_matrix2d",
    ):
        if not hasattr(kornia, name):
            setattr(kornia, name, getattr(transforms, name))


def _patch_legacy_numpy_api() -> None:
    """Keep PIMoG's Moire formula compatible with NumPy 2 and make it faster."""
    import Noise_Layer  # type: ignore

    def moire_gen(p_size: int, theta: float, center_x: float, center_y: float) -> np.ndarray:
        center_x = float(np.asarray(center_x).reshape(-1)[0])
        center_y = float(np.asarray(center_y).reshape(-1)[0])
        rows, cols = np.meshgrid(
            np.arange(1, p_size + 1), np.arange(1, p_size + 1), indexing="ij"
        )
        z1 = 0.5 + 0.5 * np.cos(
            2 * np.pi * np.sqrt((rows - center_x) ** 2 + (cols - center_y) ** 2)
        )
        radians = float(theta) / 180.0 * np.pi
        z2 = 0.5 + 0.5 * np.cos(np.cos(radians) * cols + np.sin(radians) * rows)
        return (np.minimum(z1, z2) + 1.0) / 2.0

    Noise_Layer.MoireGen = moire_gen


def load_pimog(repo_dir: Path, device: torch.device) -> nn.Module:
    """Load the official pretrained ScreenShooting network on CPU or GPU."""
    _patch_legacy_kornia_api()
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from model import Encoder_Decoder  # type: ignore

    _patch_legacy_numpy_api()
    model = Encoder_Decoder("ScreenShooting")
    weight_path = (
        repo_dir / "models" / "ScreenShooting" / "Encoder_Decoder_Model_mask_99.pth"
    )
    state = torch.load(weight_path, map_location=device, weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


@torch.no_grad()
def embed(model: nn.Module, images: Tensor, messages: Tensor) -> Tensor:
    return model.Encoder(images, messages)


@torch.no_grad()
def screen_shoot(model: nn.Module, encoded_images: Tensor) -> Tensor:
    return model.Noiser(encoded_images).float()


@torch.no_grad()
def decoder_features(decoder: nn.Module, images: Tensor) -> Tensor:
    """Return the 256-dimensional vector before PIMoG's final bit head."""
    out = decoder.layer1(images)
    extractor = decoder.extractor
    out = extractor.layer1(out)
    out = extractor.layer2(out)
    out = extractor.layer3(out)
    out = extractor.layer4(out)
    out = extractor.layer5(out)
    return out.squeeze(1).flatten(1)


@torch.no_grad()
def original_logits(decoder: nn.Module, features: Tensor) -> Tensor:
    return decoder.extractor.linear(features)
