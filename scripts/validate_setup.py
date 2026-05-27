from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pmfa.data import list_images
from pmfa.pimog import load_pimog


def count_images(directory: Path) -> int:
    return len(list_images(directory)) if directory.exists() else 0


def main() -> None:
    model_dir = PROJECT_ROOT / "third_party" / "PIMoG"
    weight = model_dir / "models" / "ScreenShooting" / "Encoder_Decoder_Model_mask_99.pth"
    if not weight.exists():
        raise FileNotFoundError(f"Missing pretrained PIMoG weight: {weight}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pimog(model_dir, device)
    with torch.no_grad():
        image = torch.zeros(1, 3, 128, 128, device=device)
        message = torch.zeros(1, 30, device=device)
        encoded = model.Encoder(image, message)
        decoded = model.Decoder(encoded)
    div2k_count = count_images(PROJECT_ROOT / "data" / "raw" / "DIV2K_valid_HR")
    kodak_count = count_images(PROJECT_ROOT / "data" / "raw" / "kodak")
    print(f"PIMoG weight: OK ({weight.name})")
    print(f"Device: {device}")
    print(f"Forward check: encoded={tuple(encoded.shape)}, decoded={tuple(decoded.shape)}")
    print(f"DIV2K images: {div2k_count}/100")
    print(f"Kodak images: {kodak_count}/24")
    if div2k_count >= 100 and kodak_count >= 24:
        print("Formal experiment readiness: READY")
    else:
        print("Formal experiment readiness: WAITING FOR DATA")


if __name__ == "__main__":
    main()

