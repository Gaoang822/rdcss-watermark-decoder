from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmfa.data import list_images
from pmfa.experiment import build_feature_cache, test_specs
from pmfa.pimog import load_pimog


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "full"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pimog(PROJECT_ROOT / "third_party" / "PIMoG", device)
    validation_images = list_images(PROJECT_ROOT / "data" / "raw" / "DIV2K_valid_HR")[80:100]
    specs = test_specs(validation_images, seed=2026 + 120000)
    for name in ("crop50", "crop50_blur", "crop50_jpeg"):
        build_feature_cache(
            model,
            specs[name],
            output_dir,
            f"validation_{name}",
            device,
            batch_size=32,
            save_example=False,
        )
        print(f"Built validation cache: {name}")


if __name__ == "__main__":
    main()

