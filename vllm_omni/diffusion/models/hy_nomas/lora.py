# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import math
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Self

import torch
from safetensors import safe_open
from torch import nn
from vllm.model_executor.models.utils import WeightsMapper


@dataclass(frozen=True)
class HyNomasLoRA:
    directory: Path
    scale: float

    @classmethod
    def from_args(cls, path: str | list[str] | None, scale: float | list[float]) -> Self | None:
        if path is None:
            return None
        if isinstance(path, list) and len(path) == 1:
            path = path[0]
        if isinstance(scale, list) and len(scale) == 1:
            scale = scale[0]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Hy-Nomas lora_path must specify one local adapter directory.")
        if type(scale) not in (int, float) or not math.isfinite(scale):
            raise ValueError("Hy-Nomas lora_scale must be a single finite number.")
        directory = Path(path).expanduser()
        for filename in ("adapter_config.json", "adapter.safetensors"):
            if not (directory / filename).is_file():
                raise FileNotFoundError(f"Missing Hy-Nomas LoRA file: {directory / filename}")
        return cls(directory, float(scale))

    def merge_weights(
        self,
        transformer: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
        names: WeightsMapper,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        config = json.loads((self.directory / "adapter_config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("format_version") != 1:
            raise ValueError("Hy-Nomas requires a version 1 LoRA adapter_config.json.")
        rank, alpha, patterns = config.get("rank"), config.get("alpha"), config.get("targets")
        if type(rank) is not int or rank <= 0:
            raise ValueError("Hy-Nomas LoRA rank must be a positive integer.")
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Hy-Nomas LoRA alpha must be a positive finite number.")
        if not isinstance(patterns, list) or not patterns or not all(isinstance(p, str) and p for p in patterns):
            raise ValueError("Hy-Nomas LoRA targets must be a nonempty list of module names or globs.")
        scale = (alpha / rank) * self.scale
        if not math.isfinite(scale):
            raise ValueError("Hy-Nomas LoRA effective scale must be finite.")

        linears = {name: module.weight for name, module in transformer.named_modules() if isinstance(module, nn.Linear)}
        targets = {}
        for pattern in patterns:
            matching = {name: weight for name, weight in linears.items() if fnmatchcase(name, pattern)}
            if not matching:
                raise ValueError(f"Hy-Nomas LoRA target matches no DiT linear: {pattern}")
            targets.update(matching)

        expected_shapes = {}
        for name, weight in targets.items():
            out_features, in_features = weight.shape
            expected_shapes[f"{name}.lora_A.weight"] = (rank, in_features)
            expected_shapes[f"{name}.lora_B.weight"] = (out_features, rank)

        # Validate the whole adapter from the header before loading any base weights.
        with safe_open(self.directory / "adapter.safetensors", framework="pt", device="cpu") as adapter:
            keys = adapter.keys()
            tensors = {}
            for key, name in zip(keys, names.apply_list(keys), strict=True):
                if name in tensors:
                    raise ValueError(f"Duplicate Hy-Nomas LoRA tensor after normalization: {name}")
                tensors[name] = key
            missing = expected_shapes.keys() - tensors.keys()
            unexpected = tensors.keys() - expected_shapes.keys()
            if missing or unexpected:
                raise ValueError(
                    "Hy-Nomas LoRA tensors do not match adapter targets. "
                    f"Missing: {sorted(missing)[:10]}. Unexpected: {sorted(unexpected)[:10]}."
                )
            for name, shape in expected_shapes.items():
                info = adapter.get_slice(tensors[name])
                if tuple(info.get_shape()) != shape:
                    raise ValueError(f"Hy-Nomas LoRA tensor {name} has shape {info.get_shape()}, expected {shape}.")
                if info.get_dtype() not in {"F32", "F16", "BF16"}:
                    raise ValueError(f"Hy-Nomas LoRA tensor {name} must be FP32, FP16 or BF16.")

            for name, weight in weights:
                stem = name.removesuffix(".weight")
                target = targets.get(stem) if name.endswith(".weight") else None
                if target is not None and scale != 0:
                    if weight.shape != target.shape:
                        raise ValueError(f"Hy-Nomas weight {name} has shape {weight.shape}, expected {target.shape}.")
                    a = adapter.get_tensor(tensors[f"{stem}.lora_A.weight"]).to(target.device, torch.float32)
                    b = adapter.get_tensor(tensors[f"{stem}.lora_B.weight"]).to(target.device, torch.float32)
                    # Use the checkpoint precision for the input and round only after the FP32 merge.
                    merged = weight.to(target.device, torch.float32, copy=True).add_((b @ a).mul_(scale))
                    weight = merged.to(dtype=weight.dtype)
                yield name, weight
