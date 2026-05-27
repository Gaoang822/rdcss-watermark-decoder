from __future__ import annotations

import copy

import torch
from torch import Tensor, nn


class SingleViewHead(nn.Module):
    """Fine-tuned linear decoding head over one frozen PIMoG feature view."""

    def __init__(self, original_linear: nn.Linear) -> None:
        super().__init__()
        self.head = nn.Linear(256, 30)
        self.head.load_state_dict(original_linear.state_dict())

    def forward(self, features: Tensor) -> Tensor:
        return self.head(features[:, 0])


class FeatureHead(nn.Module):
    """Fine-tuned watermark bit head for already synchronized features."""

    def __init__(self, original_linear: nn.Linear) -> None:
        super().__init__()
        self.head = nn.Linear(256, 30)
        self.head.load_state_dict(original_linear.state_dict())

    def forward(self, features: Tensor) -> Tensor:
        return self.head(features)


class PMFA(nn.Module):
    """Attention-based fusion adapter for spatially aligned PIMoG feature views."""

    def __init__(self, original_linear: nn.Linear) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.fusion_logit = nn.Parameter(torch.tensor(-3.0))
        self.head = nn.Linear(256, 30)
        self.head.load_state_dict(original_linear.state_dict())

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.attention(features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        aggregated = (features * weights.unsqueeze(-1)).sum(dim=1)
        gate = torch.sigmoid(self.fusion_logit)
        fused = (1.0 - gate) * features[:, 0] + gate * aggregated
        return self.head(fused), weights

    @property
    def fusion_gate(self) -> float:
        return float(torch.sigmoid(self.fusion_logit).detach().cpu())


class DecoderTailFT(nn.Module):
    """Fine-tune only the last PIMoG extractor blocks on partial observations."""

    def __init__(self, original_decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = copy.deepcopy(original_decoder)
        for parameter in self.decoder.parameters():
            parameter.requires_grad = False
        for module in (
            self.decoder.extractor.layer3,
            self.decoder.extractor.layer4,
            self.decoder.extractor.layer5,
            self.decoder.extractor.linear,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(self, images: Tensor) -> Tensor:
        return self.decoder(images)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
