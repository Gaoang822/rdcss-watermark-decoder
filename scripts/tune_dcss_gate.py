from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    bits = logits.round().clamp(0, 1)
    matches = bits.eq(labels)
    return matches.float().mean().item(), matches.all(dim=1).float().mean().item()


def error_and_predictions(
    cache: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = cache["original_logits"]
    errors = (logits - logits.round().clamp(0, 1)).square().mean(dim=2)
    selected = errors.argmin(dim=1)
    dcss = logits[torch.arange(len(selected)), selected]
    first = logits[:, 0]
    average = logits.mean(dim=1)
    return errors.amin(dim=1), dcss, first, average


def main() -> None:
    cache_root = PROJECT_ROOT / "outputs" / "full" / "cache"
    validation = {
        condition: torch.load(cache_root / f"validation_{condition}.pt", map_location="cpu", weights_only=True)
        for condition in ("crop50", "crop50_blur", "crop50_jpeg")
    }
    thresholds = np.linspace(0.12, 0.22, 1001)
    rows = []
    for fallback_mode in ("average", "first"):
        for threshold in thresholds:
            condition_scores = []
            for condition, cache in validation.items():
                best_error, dcss, first, average = error_and_predictions(cache)
                fallback = average if fallback_mode == "average" else first
                use_dcss = best_error <= threshold
                chosen = torch.where(use_dcss.unsqueeze(1), dcss, fallback)
                bit_acc, message_acc = metrics(chosen, cache["labels"])
                condition_scores.append(bit_acc)
                rows.append(
                    {
                        "fallback_mode": fallback_mode,
                        "threshold": threshold,
                        "condition": condition,
                        "bit_accuracy": bit_acc,
                        "message_accuracy": message_acc,
                        "dcss_usage_rate": use_dcss.float().mean().item(),
                    }
                )
            rows.append(
                {
                    "fallback_mode": fallback_mode,
                    "threshold": threshold,
                    "condition": "macro_average",
                    "bit_accuracy": float(np.mean(condition_scores)),
                    "message_accuracy": float("nan"),
                    "dcss_usage_rate": float("nan"),
                }
            )
    frame = pd.DataFrame(rows)
    best = (
        frame[frame["condition"] == "macro_average"]
        .assign(fallback_priority=lambda data: data["fallback_mode"].eq("first").astype(int))
        .sort_values(
            ["bit_accuracy", "fallback_priority", "threshold"],
            ascending=[False, True, True],
        )
        .iloc[0]
    )
    threshold = float(best["threshold"])
    fallback_mode = str(best["fallback_mode"])
    frame.to_csv(PROJECT_ROOT / "outputs" / "full" / "tables" / "dcss_gate_tuning.csv", index=False)
    config = {
        "dcss_gate_threshold": threshold,
        "fallback_mode": fallback_mode,
        "selection_metric": "validation macro-average Bit Accuracy",
        "validation_conditions": ["crop50", "crop50_blur", "crop50_jpeg"],
    }
    with (PROJECT_ROOT / "outputs" / "full" / "dcss_gate.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    chosen = frame[
        frame["threshold"].eq(threshold) & frame["fallback_mode"].eq(fallback_mode)
    ]
    print(config)
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    main()
