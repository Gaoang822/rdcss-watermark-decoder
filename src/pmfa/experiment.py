from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .attacks import blind_view, multi_views, oracle_view, partial_observation
from .data import list_images, load_image, save_image
from .metrics import bit_statistics, psnr, ssim
from .models import DecoderTailFT, FeatureHead, PMFA, SingleViewHead, count_trainable_parameters
from .pimog import decoder_features, embed, load_pimog, original_logits, screen_shoot


@dataclass(frozen=True)
class SampleSpec:
    image_path: str
    image_id: str
    message_id: int
    message_seed: int
    attack_seed: int
    retain_ratio: float
    condition: str
    extra: str = "none"


TEST_CONDITIONS = (
    ("full", 1.0, "none"),
    ("crop70", 0.7, "none"),
    ("crop50", 0.5, "none"),
    ("crop50_jpeg", 0.5, "jpeg"),
    ("crop50_blur", 0.5, "blur"),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def message_from_seed(seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (30,), generator=generator, dtype=torch.float32)


def training_specs(paths: list[Path], split: str, seed: int) -> list[SampleSpec]:
    specs: list[SampleSpec] = []
    for image_index, path in enumerate(paths):
        for message_id in range(3):
            message_seed = seed + image_index * 101 + message_id * 17
            for repeat in range(3):
                attack_seed = seed + image_index * 1009 + message_id * 97 + repeat
                retain_ratio = random.Random(attack_seed).uniform(0.5, 1.0)
                specs.append(
                    SampleSpec(
                        str(path),
                        path.stem,
                        message_id,
                        message_seed,
                        attack_seed,
                        retain_ratio,
                        split,
                    )
                )
    return specs


def test_specs(paths: list[Path], seed: int) -> dict[str, list[SampleSpec]]:
    outputs: dict[str, list[SampleSpec]] = {}
    for condition, retain_ratio, extra in TEST_CONDITIONS:
        specs: list[SampleSpec] = []
        for image_index, path in enumerate(paths):
            for message_id in range(2):
                message_seed = seed + image_index * 131 + message_id * 19
                for repeat in range(3):
                    attack_seed = seed + image_index * 1543 + message_id * 113 + repeat
                    specs.append(
                        SampleSpec(
                            str(path),
                            path.stem,
                            message_id,
                            message_seed,
                            attack_seed,
                            retain_ratio,
                            condition,
                            extra,
                        )
                    )
        outputs[condition] = specs
    return outputs


def build_feature_cache(
    model: nn.Module,
    specs: list[SampleSpec],
    output_dir: Path,
    name: str,
    device: torch.device,
    batch_size: int,
    save_example: bool = False,
) -> Path:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = cache_dir / f"{name}.pt"
    metadata_path = cache_dir / f"{name}_metadata.csv"
    features_all: list[Tensor] = []
    logits_all: list[Tensor] = []
    blind_logits_all: list[Tensor] = []
    oracle_logits_all: list[Tensor] = []
    labels_all: list[Tensor] = []
    direct_views_all: list[Tensor] = []
    oracle_candidate_indices: list[int] = []
    metadata: list[dict[str, object]] = []
    example_saved = False

    for start in tqdm(range(0, len(specs), batch_size), desc=f"features:{name}"):
        batch_specs = specs[start : start + batch_size]
        images = torch.stack([load_image(Path(spec.image_path)) for spec in batch_specs]).to(device)
        labels = torch.stack([message_from_seed(spec.message_seed) for spec in batch_specs]).to(device)
        seed_everything(batch_specs[0].attack_seed)
        encoded = embed(model, images, labels)
        attacked = screen_shoot(model, encoded)

        view_batches: list[Tensor] = []
        blind_batches: list[Tensor] = []
        oracle_batches: list[Tensor] = []
        position_errors: list[tuple[int, int]] = []
        for item, spec in zip(attacked, batch_specs):
            observation = partial_observation(
                item, spec.retain_ratio, spec.attack_seed, spec.extra
            )
            views, error, oracle_index = multi_views(observation, spec.attack_seed)
            view_batches.append(views)
            blind_batches.append(blind_view(observation))
            oracle_batches.append(oracle_view(observation))
            position_errors.append(error)
            oracle_candidate_indices.append(oracle_index)
        views = torch.stack(view_batches).to(device)
        blind_images = torch.stack(blind_batches).to(device)
        oracle_images = torch.stack(oracle_batches).to(device)
        flat_views = views.flatten(0, 1)
        features = decoder_features(model.Decoder, flat_views).reshape(-1, 9, 256)
        logits = original_logits(model.Decoder, features.flatten(0, 1)).reshape(-1, 9, 30)
        blind_logits = original_logits(
            model.Decoder, decoder_features(model.Decoder, blind_images)
        )
        oracle_logits = original_logits(
            model.Decoder, decoder_features(model.Decoder, oracle_images)
        )

        features_all.append(features.cpu())
        logits_all.append(logits.cpu())
        blind_logits_all.append(blind_logits.cpu())
        oracle_logits_all.append(oracle_logits.cpu())
        labels_all.append(labels.cpu())
        direct_views_all.append(views[:, 0].cpu().half())
        for index, spec in enumerate(batch_specs):
            row = asdict(spec)
            row["position_error_x"] = position_errors[index][0]
            row["position_error_y"] = position_errors[index][1]
            row["oracle_candidate_index"] = oracle_candidate_indices[-len(batch_specs) + index]
            row["message"] = "".join(str(int(bit)) for bit in labels[index].cpu())
            row["psnr"] = psnr(images[index], encoded[index])
            row["ssim"] = ssim(images[index], encoded[index])
            metadata.append(row)
        if save_example and not example_saved:
            illustration = torch.cat((images[0], encoded[0], blind_images[0], views[0, 0]), dim=2)
            save_image(illustration, output_dir / "figures" / f"example_{name}.png")
            example_saved = True

    payload = {
        "features": torch.cat(features_all),
        "original_logits": torch.cat(logits_all),
        "blind_logits": torch.cat(blind_logits_all),
        "oracle_logits": torch.cat(oracle_logits_all),
        "labels": torch.cat(labels_all),
        "direct_views": torch.cat(direct_views_all),
        "oracle_candidate_indices": torch.tensor(oracle_candidate_indices, dtype=torch.long),
    }
    torch.save(payload, tensor_path)
    pd.DataFrame(metadata).to_csv(metadata_path, index=False, encoding="utf-8-sig")
    return tensor_path


def _predict(model: nn.Module, features: Tensor) -> Tensor:
    output = model(features)
    return output[0] if isinstance(output, tuple) else output


def confidence_sync_indices(original_logits: Tensor) -> Tensor:
    """Select the alignment candidate whose deep output is nearest to a binary message."""
    quantization_error = (
        original_logits - original_logits.round().clamp(0, 1)
    ).square().mean(dim=2)
    return quantization_error.argmin(dim=1)


def reliable_dcss_logits(
    original_candidate_logits: Tensor,
    threshold: float,
    fallback_mode: str = "average",
) -> tuple[Tensor, Tensor]:
    """Use synchronized decoding only when validation-calibrated confidence allows it."""
    quantization_error = (
        original_candidate_logits - original_candidate_logits.round().clamp(0, 1)
    ).square().mean(dim=2)
    indices = quantization_error.argmin(dim=1)
    selected = original_candidate_logits[
        torch.arange(len(indices), device=original_candidate_logits.device), indices
    ]
    if fallback_mode == "average":
        fallback = original_candidate_logits.mean(dim=1)
    elif fallback_mode == "first":
        fallback = original_candidate_logits[:, 0]
    else:
        raise ValueError(f"Unsupported R-DCSS fallback mode: {fallback_mode}")
    use_dcss = quantization_error.amin(dim=1) <= threshold
    return torch.where(use_dcss.unsqueeze(1), selected, fallback), use_dcss


def train_head(
    model: nn.Module,
    train_cache: dict[str, Tensor],
    val_cache: dict[str, Tensor],
    output_dir: Path,
    name: str,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, float]:
    model.to(device)
    loader = DataLoader(
        TensorDataset(train_cache["features"], train_cache["labels"]),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_accuracy = -1.0
    remaining_patience = patience
    checkpoint = output_dir / "models" / f"{name}_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            predictions = _predict(model, features)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * features.shape[0]
        model.eval()
        with torch.no_grad():
            val_predictions = _predict(model, val_cache["features"].to(device)).cpu()
        val_stats = bit_statistics(val_predictions, val_cache["labels"])
        epoch_loss = loss_total / len(loader.dataset)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_bit_accuracy": val_stats["bit_accuracy"],
            }
        )
        if val_stats["bit_accuracy"] > best_accuracy:
            best_accuracy = val_stats["bit_accuracy"]
            remaining_patience = patience
            torch.save(model.state_dict(), checkpoint)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    train_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    history_frame = pd.DataFrame(history)
    history_frame["model"] = name
    history_frame.to_csv(output_dir / "tables" / f"{name}_history.csv", index=False)
    return model, history_frame, train_seconds


def train_pmfa(
    model: PMFA,
    train_cache: dict[str, Tensor],
    val_cache: dict[str, Tensor],
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    alignment_weight: float = 0.2,
) -> tuple[PMFA, pd.DataFrame, float]:
    model.to(device)
    loader = DataLoader(
        TensorDataset(
            train_cache["features"],
            train_cache["labels"],
            train_cache["oracle_candidate_indices"],
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_accuracy = -1.0
    best_alignment = -1.0
    remaining_patience = patience
    checkpoint = output_dir / "models" / "PMFA_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for features, targets, candidate_indices in loader:
            features = features.to(device)
            targets = targets.to(device)
            candidate_indices = candidate_indices.to(device)
            predictions, weights = model(features)
            message_loss = criterion(predictions, targets)
            alignment_loss = F.nll_loss(torch.log(weights.clamp_min(1e-8)), candidate_indices)
            loss = message_loss + alignment_weight * alignment_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * features.shape[0]
        model.eval()
        with torch.no_grad():
            val_predictions, val_weights = model(val_cache["features"].to(device))
        val_statistics = bit_statistics(val_predictions.cpu(), val_cache["labels"])
        val_alignment = (
            val_weights.argmax(dim=1).cpu() == val_cache["oracle_candidate_indices"]
        ).float().mean().item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / len(loader.dataset),
                "val_bit_accuracy": val_statistics["bit_accuracy"],
                "val_alignment_accuracy": val_alignment,
                "model": "PMFA",
            }
        )
        improved = val_statistics["bit_accuracy"] > best_accuracy or (
            val_statistics["bit_accuracy"] == best_accuracy and val_alignment > best_alignment
        )
        if improved:
            best_accuracy = val_statistics["bit_accuracy"]
            best_alignment = val_alignment
            remaining_patience = patience
            torch.save(model.state_dict(), checkpoint)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    train_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "tables" / "PMFA_history.csv", index=False)
    return model, history_frame, train_seconds


def train_tail_decoder(
    model: DecoderTailFT,
    train_cache: dict[str, Tensor],
    val_cache: dict[str, Tensor],
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[DecoderTailFT, pd.DataFrame, float]:
    model.to(device)
    loader = DataLoader(
        TensorDataset(train_cache["direct_views"], train_cache["labels"]),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4
    )
    criterion = nn.MSELoss()
    best_accuracy = -1.0
    remaining_patience = patience
    checkpoint = output_dir / "models" / "Decoder-Tail-FT_best.pt"
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for images, targets in loader:
            images, targets = images.float().to(device), targets.to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * images.shape[0]
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_cache["direct_views"].float().to(device)).cpu()
        val_accuracy = bit_statistics(val_predictions, val_cache["labels"])["bit_accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / len(loader.dataset),
                "val_bit_accuracy": val_accuracy,
                "model": "Decoder-Tail-FT",
            }
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            remaining_patience = patience
            torch.save(model.state_dict(), checkpoint)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    train_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "tables" / "Decoder-Tail-FT_history.csv", index=False)
    return model, history_frame, train_seconds


def synchronized_features(cache: dict[str, Tensor], oracle: bool) -> Tensor:
    indices = (
        cache["oracle_candidate_indices"]
        if oracle
        else confidence_sync_indices(cache["original_logits"])
    )
    return cache["features"][torch.arange(len(indices)), indices]


def train_synchronized_head(
    model: FeatureHead,
    train_cache: dict[str, Tensor],
    val_cache: dict[str, Tensor],
    output_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[FeatureHead, pd.DataFrame, float]:
    """Train on oracle-aligned simulation samples; validate using DCSS-selected samples."""
    model.to(device)
    train_features = synchronized_features(train_cache, oracle=True)
    val_features = synchronized_features(val_cache, oracle=False)
    loader = DataLoader(
        TensorDataset(train_features, train_cache["labels"]),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_accuracy = -1.0
    remaining_patience = patience
    checkpoint = output_dir / "models" / "DCSS-FT_best.pt"
    history: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for features, targets in loader:
            predictions = model(features.to(device))
            loss = criterion(predictions, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * features.shape[0]
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_features.to(device)).cpu()
        val_accuracy = bit_statistics(val_predictions, val_cache["labels"])["bit_accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / len(loader.dataset),
                "val_bit_accuracy": val_accuracy,
                "model": "DCSS-FT",
            }
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            remaining_patience = patience
            torch.save(model.state_dict(), checkpoint)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    train_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "tables" / "DCSS-FT_history.csv", index=False)
    return model, history_frame, train_seconds


def evaluate_methods(
    caches: dict[str, dict[str, Tensor]],
    single_view: SingleViewHead,
    pmfa: PMFA,
    tail_decoder: DecoderTailFT,
    dcss_head: FeatureHead,
    output_dir: Path,
    device: torch.device,
    gate_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    single_view.eval().to(device)
    pmfa.eval().to(device)
    tail_decoder.eval().to(device)
    dcss_head.eval().to(device)
    for condition, cache in caches.items():
        features = cache["features"].to(device)
        labels = cache["labels"]
        logits = cache["original_logits"]
        with torch.no_grad():
            dcss_indices = confidence_sync_indices(logits)
            dcss_features = features[torch.arange(len(dcss_indices), device=device), dcss_indices.to(device)]
            dcss_logits = logits[torch.arange(len(dcss_indices)), dcss_indices]
            reliable_logits = None
            reliable_usage = None
            if gate_config is not None:
                reliable_logits, reliable_usage = reliable_dcss_logits(
                    logits,
                    float(gate_config["dcss_gate_threshold"]),
                    str(gate_config.get("fallback_mode", "average")),
                )
            pmfa_logits, pmfa_weights = pmfa(features)
            predictions = {
                "PIMoG-Original": cache["blind_logits"],
                "Coarse-Prior": logits[:, 0],
                "Multi-View-Average": logits.mean(dim=1),
                "Single-View-FT": single_view(features).cpu(),
                "PMFA": pmfa_logits.cpu(),
                "Decoder-Tail-FT": tail_decoder(cache["direct_views"].float().to(device)).cpu(),
                "Oracle-Alignment": cache["oracle_logits"],
                "DCSS": dcss_logits,
                "DCSS-FT": dcss_head(dcss_features).cpu(),
            }
            if reliable_logits is not None:
                predictions["R-DCSS"] = reliable_logits
        for method, method_logits in predictions.items():
            bits = method_logits.round().clamp(0, 1)
            correct = bits.eq(labels)
            for item in range(labels.shape[0]):
                bit_accuracy = correct[item].float().mean().item()
                records.append(
                    {
                        "condition": condition,
                        "method": method,
                        "sample": item,
                        "bit_accuracy": bit_accuracy,
                        "ber": 1.0 - bit_accuracy,
                        "message_accuracy": float(correct[item].all()),
                        "selected_oracle_candidate": (
                            float(pmfa_weights[item].argmax().cpu() == cache["oracle_candidate_indices"][item])
                            if method == "PMFA"
                            else np.nan
                        ),
                        "dcss_usage_rate": (
                            float(reliable_usage[item]) if method == "R-DCSS" else np.nan
                        ),
                    }
                )
    raw = pd.DataFrame(records)
    raw.to_csv(output_dir / "tables" / "per_sample_results.csv", index=False)
    summary = (
        raw.groupby(["condition", "method"], as_index=False, dropna=False)[
            ["bit_accuracy", "ber", "message_accuracy", "selected_oracle_candidate", "dcss_usage_rate"]
        ]
        .mean()
        .sort_values(["condition", "method"])
    )
    summary.to_csv(output_dir / "tables" / "summary_results.csv", index=False)
    return summary


def benchmark_decoding(
    backbone: nn.Module,
    specs: list[SampleSpec],
    single_view: SingleViewHead,
    pmfa: PMFA,
    tail_decoder: DecoderTailFT,
    dcss_head: FeatureHead,
    output_dir: Path,
    device: torch.device,
    gate_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Measure decoding from already-captured partial observations to message output."""
    benchmark_specs = specs[: min(24, len(specs))]
    images = torch.stack([load_image(Path(spec.image_path)) for spec in benchmark_specs]).to(device)
    labels = torch.stack([message_from_seed(spec.message_seed) for spec in benchmark_specs]).to(device)
    seed_everything(benchmark_specs[0].attack_seed)
    encoded = embed(backbone, images, labels)
    attacked = screen_shoot(backbone, encoded)
    observations = [
        partial_observation(item, spec.retain_ratio, spec.attack_seed, spec.extra)
        for item, spec in zip(attacked, benchmark_specs)
    ]
    blind_images = torch.stack([blind_view(observation) for observation in observations]).to(device)
    views = torch.stack(
        [multi_views(observation, spec.attack_seed)[0] for observation, spec in zip(observations, benchmark_specs)]
    ).to(device)
    single_view.eval().to(device)
    pmfa.eval().to(device)
    tail_decoder.eval().to(device)
    dcss_head.eval().to(device)

    def original() -> Tensor:
        features = decoder_features(backbone.Decoder, blind_images)
        return original_logits(backbone.Decoder, features)

    def coarse_prior() -> Tensor:
        features = decoder_features(backbone.Decoder, views[:, 0])
        return original_logits(backbone.Decoder, features)

    def average() -> Tensor:
        features = decoder_features(backbone.Decoder, views.flatten(0, 1)).reshape(-1, 9, 256)
        return original_logits(backbone.Decoder, features.flatten(0, 1)).reshape(-1, 9, 30).mean(1)

    def fitted_single() -> Tensor:
        features = decoder_features(backbone.Decoder, views[:, 0]).unsqueeze(1)
        return single_view(features)

    def fitted_pmfa() -> Tensor:
        features = decoder_features(backbone.Decoder, views.flatten(0, 1)).reshape(-1, 9, 256)
        return pmfa(features)[0]

    def fitted_tail() -> Tensor:
        return tail_decoder(views[:, 0])

    def dcss() -> Tensor:
        features = decoder_features(backbone.Decoder, views.flatten(0, 1)).reshape(-1, 9, 256)
        logits = original_logits(backbone.Decoder, features.flatten(0, 1)).reshape(-1, 9, 30)
        indices = confidence_sync_indices(logits)
        return logits[torch.arange(len(indices), device=device), indices]

    def dcss_ft() -> Tensor:
        features = decoder_features(backbone.Decoder, views.flatten(0, 1)).reshape(-1, 9, 256)
        logits = original_logits(backbone.Decoder, features.flatten(0, 1)).reshape(-1, 9, 30)
        indices = confidence_sync_indices(logits)
        selected = features[torch.arange(len(indices), device=device), indices]
        return dcss_head(selected)

    methods = {
        "PIMoG-Original": original,
        "Coarse-Prior": coarse_prior,
        "Multi-View-Average": average,
        "Single-View-FT": fitted_single,
        "PMFA": fitted_pmfa,
        "Decoder-Tail-FT": fitted_tail,
        "DCSS": dcss,
        "DCSS-FT": dcss_ft,
    }
    if gate_config is not None:
        def reliable_dcss() -> Tensor:
            features = decoder_features(backbone.Decoder, views.flatten(0, 1)).reshape(-1, 9, 256)
            logits = original_logits(backbone.Decoder, features.flatten(0, 1)).reshape(-1, 9, 30)
            return reliable_dcss_logits(
                logits,
                float(gate_config["dcss_gate_threshold"]),
                str(gate_config.get("fallback_mode", "average")),
            )[0]

        methods["R-DCSS"] = reliable_dcss
    records: list[dict[str, float | str]] = []
    with torch.no_grad():
        for method, call in methods.items():
            call()
            durations = []
            for _ in range(3):
                started = time.perf_counter()
                call()
                durations.append(time.perf_counter() - started)
            records.append(
                {
                    "method": method,
                    "decode_ms_per_sample": 1000 * float(np.mean(durations)) / len(benchmark_specs),
                }
            )
    benchmark = pd.DataFrame(records)
    benchmark.to_csv(output_dir / "tables" / "decoding_runtime.csv", index=False)
    return benchmark


def plot_results(
    histories: list[pd.DataFrame], summary: pd.DataFrame, output_dir: Path
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for history in histories:
        label = history["model"].iloc[0]
        plt.plot(history["epoch"], history["val_bit_accuracy"], marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Bit Accuracy")
    plt.ylim(0.45, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "validation_accuracy_curve.png", dpi=220)
    plt.close()

    highlighted_methods = {
        "PIMoG-Original",
        "Coarse-Prior",
        "Decoder-Tail-FT",
        "DCSS",
        "R-DCSS",
        "Oracle-Alignment",
    }
    main = summary[
        summary["condition"].isin(["full", "crop70", "crop50"])
        & summary["method"].isin(highlighted_methods)
    ].copy()
    main["retain"] = main["condition"].map({"full": 100, "crop70": 70, "crop50": 50})
    plt.figure(figsize=(7, 4))
    for method, frame in main.groupby("method"):
        frame = frame.sort_values("retain")
        plt.plot(frame["retain"], frame["bit_accuracy"], marker="o", label=method)
    plt.xlabel("Observed Image Area (%)")
    plt.ylabel("Bit Accuracy")
    plt.ylim(0.4, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "crop_robustness_curve.png", dpi=220)
    plt.close()


def run_experiment(
    project_root: Path,
    data_root: Path,
    output_dir: Path,
    seed: int = 2026,
    quick: bool = False,
    reuse_cache: bool = False,
    reuse_baseline_models: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pimog(project_root / "third_party" / "PIMoG", device)

    div2k = list_images(data_root / "raw" / "DIV2K_valid_HR")
    kodak = list_images(data_root / "raw" / "kodak")
    minimum_div2k = 6 if quick else 100
    minimum_kodak = 3 if quick else 24
    if len(div2k) < minimum_div2k or len(kodak) < minimum_kodak:
        raise ValueError(
            f"Insufficient data: need DIV2K >= {minimum_div2k} and Kodak >= {minimum_kodak}; "
            f"found DIV2K={len(div2k)}, Kodak={len(kodak)}."
        )
    if quick:
        train_paths, val_paths, test_paths = div2k[:4], div2k[4:6], kodak[:3]
    else:
        train_paths, val_paths, test_paths = div2k[:80], div2k[80:100], kodak[:24]

    train_sample_specs = training_specs(train_paths, "train_random", seed)
    val_sample_specs = training_specs(val_paths, "validation_random", seed + 50000)
    test_spec_map = test_specs(test_paths, seed + 90000)
    batch_size = 8 if quick else 32
    def cached_or_build(name: str, specs: list[SampleSpec], save_example: bool) -> Path:
        path = output_dir / "cache" / f"{name}.pt"
        if reuse_cache and path.exists():
            return path
        return build_feature_cache(
            model, specs, output_dir, name, device, batch_size, save_example=save_example
        )

    train_path = cached_or_build("train", train_sample_specs, False)
    val_path = cached_or_build("validation", val_sample_specs, False)
    test_paths_map: dict[str, Path] = {}
    for condition, specs in test_spec_map.items():
        test_paths_map[condition] = cached_or_build(condition, specs, True)

    train_cache = torch.load(train_path, map_location="cpu", weights_only=True)
    val_cache = torch.load(val_path, map_location="cpu", weights_only=True)
    original_linear = model.Decoder.extractor.linear
    max_epochs = 30
    prior_info_path = output_dir / "run_info.json"
    prior_info = {}
    if prior_info_path.exists():
        with prior_info_path.open(encoding="utf-8") as handle:
            prior_info = json.load(handle)
    if reuse_baseline_models and all(
        (output_dir / "models" / name).exists()
        for name in ("Single-View-FT_best.pt", "PMFA_best.pt", "Decoder-Tail-FT_best.pt")
    ):
        single = SingleViewHead(original_linear).to(device)
        single.load_state_dict(
            torch.load(output_dir / "models" / "Single-View-FT_best.pt", map_location=device, weights_only=True)
        )
        pmfa = PMFA(original_linear).to(device)
        pmfa.load_state_dict(
            torch.load(output_dir / "models" / "PMFA_best.pt", map_location=device, weights_only=True)
        )
        tail = DecoderTailFT(model.Decoder).to(device)
        tail.load_state_dict(
            torch.load(output_dir / "models" / "Decoder-Tail-FT_best.pt", map_location=device, weights_only=True)
        )
        single_history = pd.read_csv(output_dir / "tables" / "Single-View-FT_history.csv")
        pmfa_history = pd.read_csv(output_dir / "tables" / "PMFA_history.csv")
        tail_history = pd.read_csv(output_dir / "tables" / "Decoder-Tail-FT_history.csv")
        single_seconds = float(prior_info.get("single_view_train_seconds", 0.0))
        pmfa_seconds = float(prior_info.get("pmfa_train_seconds", 0.0))
        tail_seconds = float(prior_info.get("tail_decoder_train_seconds", 0.0))
    else:
        single, single_history, single_seconds = train_head(
            SingleViewHead(original_linear),
            train_cache,
            val_cache,
            output_dir,
            "Single-View-FT",
            max_epochs,
            5,
            32,
            device,
        )
        pmfa, pmfa_history, pmfa_seconds = train_pmfa(
            PMFA(original_linear),
            train_cache,
            val_cache,
            output_dir,
            max_epochs,
            10 if quick else 5,
            32,
            device,
        )
        tail, tail_history, tail_seconds = train_tail_decoder(
            DecoderTailFT(model.Decoder),
            train_cache,
            val_cache,
            output_dir,
            8 if quick else 10,
            5,
            16 if quick else 32,
            device,
        )
    dcss_ft, dcss_history, dcss_seconds = train_synchronized_head(
        FeatureHead(original_linear),
        train_cache,
        val_cache,
        output_dir,
        max_epochs,
        5,
        32,
        device,
    )
    caches = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in test_paths_map.items()
    }
    gate_path = output_dir / "dcss_gate.json"
    gate_config = None
    if gate_path.exists():
        with gate_path.open(encoding="utf-8") as handle:
            gate_config = json.load(handle)
    summary = evaluate_methods(
        caches, single, pmfa, tail, dcss_ft, output_dir, device, gate_config=gate_config
    )
    runtime = benchmark_decoding(
        model,
        test_spec_map["crop50"],
        single,
        pmfa,
        tail,
        dcss_ft,
        output_dir,
        device,
        gate_config=gate_config,
    )
    plot_results([single_history, pmfa_history, tail_history, dcss_history], summary, output_dir)

    quality = pd.read_csv(output_dir / "cache" / "full_metadata.csv")
    quality = quality.drop_duplicates(["image_id", "message_id"])[["psnr", "ssim"]]
    quality_summary = {
        "psnr_mean": float(quality["psnr"].mean()),
        "ssim_mean": float(quality["ssim"].mean()),
    }
    run_info: dict[str, object] = {
        "seed": seed,
        "device": str(device),
        "quick": quick,
        "train_samples": len(train_sample_specs),
        "validation_samples": len(val_sample_specs),
        "test_samples_per_condition": len(next(iter(test_spec_map.values()))),
        "single_view_parameters": count_trainable_parameters(single),
        "pmfa_parameters": count_trainable_parameters(pmfa),
        "pmfa_fusion_gate": pmfa.fusion_gate,
        "tail_decoder_parameters": count_trainable_parameters(tail),
        "dcss_ft_parameters": count_trainable_parameters(dcss_ft),
        "single_view_train_seconds": single_seconds,
        "pmfa_train_seconds": pmfa_seconds,
        "tail_decoder_train_seconds": tail_seconds,
        "dcss_ft_train_seconds": dcss_seconds,
        "r_dcss_gate": gate_config,
        "runtime_table": runtime.to_dict(orient="records"),
        **quality_summary,
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2)
    return {"run_info": run_info, "summary": summary}
