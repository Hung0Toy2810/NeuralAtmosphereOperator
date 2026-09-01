"""Operator-runtime metadata and checkpoint compatibility guards."""

from __future__ import annotations

import hashlib
import platform
import re
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch


OPERATOR_SOURCE = "torch_harmonics.examples.models.sfno"
SUPPORTED_TORCH_HARMONICS = (0, 7, 4)


def source_tree_sha256() -> str:
    """Hash executable project sources that can change training numerics."""
    root = Path(__file__).resolve().parents[3]
    files = [
        path
        for directory in ("src", "configs", "scripts", "data")
        for path in (root / directory).rglob("*.py")
    ]
    files.extend(
        path
        for name in ("pyproject.toml", "requirements.txt", "requirements-lock.txt")
        if (path := root / name).is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_torch_harmonics() -> tuple[str, str]:
    for distribution in (
        "torch-harmonics-cu126",
        "torch-harmonics-cu128",
        "torch-harmonics-cu129",
        "torch-harmonics-cu130",
        "torch-harmonics",
    ):
        try:
            return distribution, version(distribution)
        except PackageNotFoundError:
            continue
    raise RuntimeError("No torch-harmonics distribution is installed")


def _numeric_version(value: str) -> tuple[int, ...]:
    pieces: list[int] = []
    for piece in value.split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        pieces.append(int(match.group()))
    return tuple(pieces)


def runtime_fingerprint(device: torch.device) -> dict[str, Any]:
    """Describe the exact spherical-operator runtime producing numerics."""
    distribution, harmonics_version = _installed_torch_harmonics()
    sfno_module = import_module(OPERATOR_SOURCE)
    implementation = (
        "SphericalFourierNeuralOperator"
        if hasattr(sfno_module, "SphericalFourierNeuralOperator")
        else "SphericalFourierNeuralOperatorNet"
    )
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_harmonics_distribution": distribution,
        "torch_harmonics": harmonics_version,
        "operator_source": OPERATOR_SOURCE,
        "operator_implementation": implementation,
        "source_sha256": source_tree_sha256(),
        "device_type": device.type,
        "cuda_runtime": torch.version.cuda,
    }


def validate_supported_runtime(device: torch.device) -> dict[str, Any]:
    """Validate backend availability and the exact audited SFNO implementation."""
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was selected but is unavailable")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device type: {device.type}")
    actual = runtime_fingerprint(device)
    version_tuple = _numeric_version(str(actual["torch_harmonics"]))
    if version_tuple != SUPPORTED_TORCH_HARMONICS:
        supported = ".".join(map(str, SUPPORTED_TORCH_HARMONICS))
        raise RuntimeError(
            f"This model is audited for torch-harmonics=={supported}; received "
            f"{actual['torch_harmonics']}"
        )
    return actual


def checkpoint_runtime_mismatches(
    saved: dict[str, Any], current: dict[str, Any]
) -> tuple[str, ...]:
    """Return runtime fields capable of changing SFNO numerics."""
    keys = (
        "torch",
        "torch_harmonics_distribution",
        "torch_harmonics",
        "operator_source",
        "operator_implementation",
        "source_sha256",
        "device_type",
        "cuda_runtime",
    )
    return tuple(key for key in keys if saved.get(key) != current.get(key))


def enforce_checkpoint_runtime(
    checkpoint: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_mismatch: bool = False,
) -> tuple[str, ...]:
    """Reject numerically non-equivalent resume/evaluation by default."""
    saved = checkpoint.get("runtime")
    if saved is None:
        if not allow_mismatch:
            raise RuntimeError(
                "Checkpoint has no operator-runtime metadata. Use the mismatch "
                "override only for diagnostic weight initialization."
            )
        return ("runtime_metadata_missing",)
    mismatches = checkpoint_runtime_mismatches(saved, current)
    if mismatches and not allow_mismatch:
        raise RuntimeError(
            "Checkpoint operator runtime differs in: "
            f"{', '.join(mismatches)}. Exact continuation is unsafe."
        )
    return mismatches
