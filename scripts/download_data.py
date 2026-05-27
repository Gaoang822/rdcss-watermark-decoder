from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmfa.data import download_public_data, list_images


def main() -> None:
    # 仅获取 Kodak 数据集路径
    kodak = download_public_data(PROJECT_ROOT / "data")
    print(f"Kodak test images: {len(list_images(kodak))} in {kodak}")


if __name__ == "__main__":
    main()