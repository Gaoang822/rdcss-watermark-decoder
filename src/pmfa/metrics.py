from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch import Tensor


def decoded_bits(logits: Tensor) -> Tensor:
    return logits.round().clamp(0, 1)


def bit_statistics(logits: Tensor, targets: Tensor) -> dict[str, float]:
    predictions = decoded_bits(logits)
    correct = predictions.eq(targets)
    bit_accuracy = correct.float().mean().item()
    message_accuracy = correct.all(dim=1).float().mean().item()
    return {
        "bit_accuracy": bit_accuracy,
        "ber": 1.0 - bit_accuracy,
        "message_accuracy": message_accuracy,
    }


def psnr(original: Tensor, watermarked: Tensor) -> float:
    original_01 = (original.detach().cpu().numpy() + 1.0) / 2.0
    watermarked_01 = (watermarked.detach().cpu().numpy() + 1.0) / 2.0
    mse = np.mean((original_01 - watermarked_01) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def ssim(original: Tensor, watermarked: Tensor) -> float:
    original_01 = ((original.detach().cpu().numpy() + 1.0) / 2.0).transpose(1, 2, 0)
    watermarked_01 = ((watermarked.detach().cpu().numpy() + 1.0) / 2.0).transpose(
        1, 2, 0
    )
    return float(structural_similarity(original_01, watermarked_01, channel_axis=2, data_range=1.0))
