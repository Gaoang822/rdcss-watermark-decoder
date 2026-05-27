from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def bit_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    bits = logits.round().clamp(0, 1)
    return bits.eq(labels).float().mean().item()


def evaluate(cache_path: Path) -> list[dict[str, float | str]]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    logits = cache["original_logits"]
    labels = cache["labels"]
    indices = cache["oracle_candidate_indices"]
    rows: list[dict[str, float | str]] = []
    selectors = {
        "first": torch.zeros(len(labels), dtype=torch.long),
        "confidence_mean": (logits - 0.5).abs().mean(dim=2).argmax(dim=1),
        "confidence_min": (logits - 0.5).abs().amin(dim=2).argmax(dim=1),
        "confidence_l2": ((logits - logits.round().clamp(0, 1)) ** 2).mean(dim=2).argmin(dim=1),
        "oracle_index": indices,
    }
    for name, chosen in selectors.items():
        selected = logits[torch.arange(len(labels)), chosen]
        rows.append(
            {
                "selector": name,
                "bit_accuracy": bit_accuracy(selected, labels),
                "selection_accuracy": chosen.eq(indices).float().mean().item(),
            }
        )
    rounded = logits.round().clamp(0, 1)
    majority = (rounded.mean(dim=1) >= 0.5).float()
    weights = torch.softmax((logits - 0.5).abs().mean(dim=2) * 10, dim=1)
    soft = (logits * weights.unsqueeze(-1)).sum(dim=1)
    rows.append(
        {
            "selector": "majority_bits",
            "bit_accuracy": majority.eq(labels).float().mean().item(),
            "selection_accuracy": float("nan"),
        }
    )
    rows.append(
        {
            "selector": "confidence_soft_fusion",
            "bit_accuracy": bit_accuracy(soft, labels),
            "selection_accuracy": float("nan"),
        }
    )
    return rows


def main() -> None:
    result_root = PROJECT_ROOT / "outputs" / "full" / "cache"
    frames = []
    for split in ("validation", "crop70", "crop50", "crop50_jpeg", "crop50_blur"):
        path = result_root / f"{split}.pt"
        if path.exists():
            frame = pd.DataFrame(evaluate(path))
            frame.insert(0, "split", split)
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    print(result.to_string(index=False))
    result.to_csv(PROJECT_ROOT / "outputs" / "full" / "tables" / "candidate_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
