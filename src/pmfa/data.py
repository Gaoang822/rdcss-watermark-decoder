from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

import requests
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 仅保留 Kodak 下载地址
KODAK_BASE_URL = "https://r0k.us/graphics/kodak/kodak/"


def _download(urls: str | tuple[str, ...], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    candidates = (urls,) if isinstance(urls, str) else urls
    last_error: Exception | None = None
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        for url in candidates:
            try:
                with requests.get(url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    with temporary.open("wb") as handle, tqdm(
                        total=total, unit="B", unit_scale=True, desc=destination.name
                    ) as progress:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                progress.update(len(chunk))
                temporary.replace(destination)
                return
            except requests.RequestException as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                time.sleep(attempt + 1)
    raise RuntimeError(f"Unable to download {destination.name}") from last_error


def download_public_data(data_root: Path) -> Path:
    """【仅下载】Kodak test set."""
    raw_root = data_root / "raw"
    # 仅保留 Kodak 下载逻辑
    kodak_dir = raw_root / "kodak"
    kodak_dir.mkdir(parents=True, exist_ok=True)
    # 下载 24 张 Kodak 测试图像
    for image_id in range(1, 25):
        _download(
            f"{KODAK_BASE_URL}kodim{image_id:02d}.png",
            kodak_dir / f"kodim{image_id:02d}.png",
        )
    # 仅返回 Kodak 路径
    return kodak_dir


def list_images(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return paths


def load_image(path: Path, size: int = 128) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        image = image.crop((left, top, left + crop_size, top + crop_size))
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        values = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(values).permute(2, 0, 1) / 127.5 - 1.0


def save_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = ((tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).byte()
    array = values.permute(1, 2, 0).numpy()
    Image.fromarray(array, mode="RGB").save(path)


def clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)