from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import gaussian_blur


@dataclass(frozen=True)
class PartialObservation:
    image: Tensor
    left: int
    top: int
    side: int


def partial_observation(
    attacked: Tensor, retain_ratio: float, seed: int, extra: str = "none"
) -> PartialObservation:
    """Crop a captured image and optionally apply an unseen degradation."""
    _, height, width = attacked.shape
    side = max(24, min(height, int(round(math.sqrt(retain_ratio) * height))))
    generator = random.Random(seed)
    left = generator.randint(0, width - side)
    top = generator.randint(0, height - side)
    observation = attacked[:, top : top + side, left : left + side]
    if extra == "jpeg":
        observation = _jpeg(observation, quality=45)
    elif extra == "blur":
        observation = gaussian_blur(observation, kernel_size=[5, 5], sigma=[1.2, 1.2])
    return PartialObservation(observation, left, top, side)


def blind_view(observation: PartialObservation, output_size: int = 128) -> Tensor:
    return _resize(observation.image, output_size)


def oracle_view(observation: PartialObservation, output_size: int = 128) -> Tensor:
    return _place(observation.image, observation.left, observation.top, output_size)


def multi_views(
    observation: PartialObservation, seed: int, output_size: int = 128
) -> tuple[Tensor, tuple[int, int], int]:
    """Create nine alignment hypotheses around a noisy coarse-position prior."""
    maximum = output_size - observation.side
    if maximum <= 0:
        return torch.stack([observation.image] * 9), (0, 0), 0
    generator = random.Random(seed + 7127)
    left_errors = [error for error in (-2, 0, 2) if 0 <= observation.left + error <= maximum]
    top_errors = [error for error in (-2, 0, 2) if 0 <= observation.top + error <= maximum]
    estimated_left = observation.left + generator.choice(left_errors)
    estimated_top = observation.top + generator.choice(top_errors)
    offsets = ((0, 0), (-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2))
    placements = [
        (_limit(estimated_left + dx, maximum), _limit(estimated_top + dy, maximum))
        for dx, dy in offsets
    ]
    views = [_place(observation.image, left, top, output_size) for left, top in placements]
    oracle_index = placements.index((observation.left, observation.top))
    return torch.stack(views), (
        estimated_left - observation.left,
        estimated_top - observation.top,
    ), oracle_index


def _resize(image: Tensor, size: int) -> Tensor:
    return F.interpolate(
        image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(0)


def _place(image: Tensor, left: int, top: int, size: int) -> Tensor:
    height, width = image.shape[-2:]
    if height >= size and width >= size:
        return _resize(image, size)
    right = size - width - left
    bottom = size - height - top
    padded = F.pad(image.unsqueeze(0), (left, right, top, bottom), mode="replicate")
    return padded.squeeze(0)


def _limit(value: int, maximum: int) -> int:
    return max(0, min(maximum, value))


def _jpeg(image: Tensor, quality: int) -> Tensor:
    values = ((image.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte()
    pil = Image.fromarray(values.permute(1, 2, 0).numpy(), mode="RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    restored = Image.open(buffer).convert("RGB")
    tensor = torch.from_numpy(np.asarray(restored, dtype=np.float32).copy()).permute(2, 0, 1)
    return tensor / 127.5 - 1.0
