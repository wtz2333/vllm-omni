# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter-driven benchmark core for stateful interactive world models."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import math
import statistics
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = 1


class BenchmarkContractError(RuntimeError):
    """An adapter response violated the generic benchmark contract."""


@dataclass(frozen=True)
class TickSpec:
    """One immutable interaction update submitted at a chunk boundary."""

    event_id: int
    prompt: str | None
    controls: tuple[Mapping[str, Any], ...]
    expected_output_units: float | None = None
    label: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TickSpec:
        event_id = value.get("event_id")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
            raise ValueError("tick event_id must be a non-negative integer")
        prompt = value.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("tick prompt must be a non-empty string when provided")
        controls_value = value.get("controls", ())
        if not isinstance(controls_value, Sequence) or isinstance(controls_value, (str, bytes, bytearray)):
            raise ValueError("tick controls must be a sequence of mappings")
        controls: list[Mapping[str, Any]] = []
        for control in controls_value:
            if not isinstance(control, Mapping):
                raise ValueError("each tick control must be a mapping")
            controls.append(dict(control))
        expected_output_units = value.get("expected_output_units")
        if expected_output_units is not None:
            expected_output_units = float(expected_output_units)
            if not math.isfinite(expected_output_units) or expected_output_units <= 0:
                raise ValueError("expected_output_units must be positive and finite")
        label = value.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError("tick label must be a string")
        return cls(
            event_id=event_id,
            prompt=prompt,
            controls=tuple(controls),
            expected_output_units=expected_output_units,
            label=label,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadSpec:
    """A deterministic interactive session workload."""

    scenario_id: str
    output_rate_hz: float
    ticks: tuple[TickSpec, ...]
    metadata: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkloadSpec:
        scenario_id = value.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string")
        output_rate_hz = float(value.get("output_rate_hz", 0.0))
        if not math.isfinite(output_rate_hz) or output_rate_hz <= 0:
            raise ValueError("output_rate_hz must be positive and finite")
        raw_ticks = value.get("ticks")
        if not isinstance(raw_ticks, list) or not raw_ticks:
            raise ValueError("workload must contain a non-empty ticks list")
        ticks = tuple(TickSpec.from_dict(tick) for tick in raw_ticks)
        event_ids = tuple(tick.event_id for tick in ticks)
        if event_ids != tuple(sorted(set(event_ids))):
            raise ValueError("tick event_id values must be unique and strictly increasing")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("workload metadata must be a mapping")
        return cls(
            scenario_id=scenario_id,
            output_rate_hz=output_rate_hz,
            ticks=ticks,
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "output_rate_hz": self.output_rate_hz,
            "ticks": [tick.to_dict() for tick in self.ticks],
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class InteractiveWorldModelAdapter(Protocol):
    """Model/runtime boundary consumed by the generic benchmark."""

    async def initialize(self, workload: WorkloadSpec) -> Mapping[str, Any]:
        """Initialize the runtime and return serializable runtime metadata."""
        ...

    async def create_session(self, session_id: str, workload: WorkloadSpec) -> None:
        """Create a fresh stateful generation session."""
        ...

    async def execute_tick(
        self,
        session_id: str,
        tick_index: int,
        tick: TickSpec,
    ) -> Mapping[str, Any]:
        """Execute one interaction tick and return the observation contract."""
        ...

    async def close_session(self, session_id: str) -> None:
        """Release all state owned by one session."""
        ...

    async def shutdown(self) -> None:
        """Release process-wide runtime resources."""
        ...


def load_workload(path: Path) -> WorkloadSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported workload schema_version: {payload.get('schema_version')!r}")
    raw_workload = payload.get("workload")
    if not isinstance(raw_workload, Mapping):
        raise ValueError("workload file must contain a workload mapping")
    return WorkloadSpec.from_dict(raw_workload)


def load_adapter(spec: str, adapter_args: Mapping[str, Any]) -> InteractiveWorldModelAdapter:
    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("--adapter must use module.path:factory syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError(f"adapter factory {spec!r} is not callable")
    adapter = factory(dict(adapter_args))
    if inspect.isawaitable(adapter):
        raise TypeError("adapter factory must be synchronous and return an adapter instance")
    if not isinstance(adapter, InteractiveWorldModelAdapter):
        raise TypeError(f"adapter returned by {spec!r} does not implement the required protocol")
    return adapter


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
            "cv": None,
        }
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return {
        "count": len(samples),
        "mean": mean,
        "p50": percentile(samples, 50),
        "p90": percentile(samples, 90),
        "p95": percentile(samples, 95),
        "min": min(samples),
        "max": max(samples),
        "stdev": stdev,
        "cv": stdev / mean if mean else None,
    }


def _normalize_observation(
    response: Mapping[str, Any],
    *,
    session_id: str,
    tick_index: int,
    tick: TickSpec,
) -> dict[str, Any]:
    observed_session_id = response.get("session_id")
    if not isinstance(observed_session_id, str) or observed_session_id != session_id:
        raise BenchmarkContractError("observation session_id does not match the submitted session")
    observed_event_id = response.get("event_id")
    if isinstance(observed_event_id, bool) or not isinstance(observed_event_id, int):
        raise BenchmarkContractError("observation event_id must be an integer")
    if observed_event_id != tick.event_id:
        raise BenchmarkContractError("observation event_id does not match the submitted tick")
    observed_tick_index = response.get("tick_index")
    if isinstance(observed_tick_index, bool) or not isinstance(observed_tick_index, int):
        raise BenchmarkContractError("observation tick_index must be an integer")
    if observed_tick_index != tick_index:
        raise BenchmarkContractError("observation tick_index does not match the submitted tick")
    finite = response.get("finite")
    if finite is not True:
        raise BenchmarkContractError("observation must explicitly report finite=true")
    output_units = response.get("output_units")
    if isinstance(output_units, bool) or not isinstance(output_units, (int, float)):
        raise BenchmarkContractError("observation output_units must be numeric")
    output_units = float(output_units)
    if not math.isfinite(output_units) or output_units <= 0:
        raise BenchmarkContractError("observation output_units must be positive and finite")
    if tick.expected_output_units is not None and not math.isclose(
        output_units,
        tick.expected_output_units,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise BenchmarkContractError(
            f"observation output_units={output_units} != expected {tick.expected_output_units}"
        )
    stage_durations = response.get("stage_durations", {})
    if not isinstance(stage_durations, Mapping):
        raise BenchmarkContractError("observation stage_durations must be a mapping")
    normalized_stages: dict[str, float] = {}
    for name, duration in stage_durations.items():
        try:
            value = float(duration)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BenchmarkContractError("stage durations must be numeric") from exc
        if not math.isfinite(value) or value < 0:
            raise BenchmarkContractError("stage durations must be finite and non-negative")
        normalized_stages[str(name)] = value
    peak_memory_mb = response.get("peak_memory_mb")
    if peak_memory_mb is not None:
        try:
            peak_memory_mb = float(peak_memory_mb)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BenchmarkContractError("peak_memory_mb must be numeric") from exc
        if not math.isfinite(peak_memory_mb) or peak_memory_mb < 0:
            raise BenchmarkContractError("peak_memory_mb must be finite and non-negative")
    shape = response.get("shape")
    if shape is not None:
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes, bytearray)):
            raise BenchmarkContractError("observation shape must be a sequence")
        if any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in shape):
            raise BenchmarkContractError("observation shape dimensions must be integers")
        shape = list(shape)
        if any(dimension <= 0 for dimension in shape):
            raise BenchmarkContractError("observation shape dimensions must be positive")
    metadata = response.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise BenchmarkContractError("observation metadata must be a mapping")
    return {
        "session_id": session_id,
        "event_id": tick.event_id,
        "tick_index": tick_index,
        "finite": True,
        "output_units": output_units,
        "shape": shape,
        "stage_durations": normalized_stages,
        "peak_memory_mb": peak_memory_mb,
        "metadata": dict(metadata),
    }


def summarize_samples(samples: Sequence[Mapping[str, Any]], *, output_rate_hz: float) -> dict[str, Any]:
    measured = [sample for sample in samples if sample.get("measured")]
    tick_latencies = [float(sample["latency_seconds"]) for sample in measured]
    first_tick_latencies = [float(sample["latency_seconds"]) for sample in measured if int(sample["tick_index"]) == 0]
    steady_tick_latencies = [float(sample["latency_seconds"]) for sample in measured if int(sample["tick_index"]) > 0]
    realtime_factors = [
        float(sample["latency_seconds"]) / (float(sample["output_units"]) / output_rate_hz) for sample in measured
    ]
    stage_values: dict[str, list[float]] = defaultdict(list)
    for sample in measured:
        for name, duration in sample["stage_durations"].items():
            stage_values[str(name)].append(float(duration))
    total_units = sum(float(sample["output_units"]) for sample in measured)
    total_latency = sum(tick_latencies)
    per_tick_index: dict[str, dict[str, float | int | None]] = {}
    for tick_index in sorted({int(sample["tick_index"]) for sample in measured}):
        per_tick_index[str(tick_index)] = summarize_values(
            [float(sample["latency_seconds"]) for sample in measured if int(sample["tick_index"]) == tick_index]
        )
    per_label: dict[str, dict[str, float | int | None]] = {}
    for label in sorted({str(sample["label"]) for sample in measured if sample.get("label") is not None}):
        per_label[label] = summarize_values(
            [float(sample["latency_seconds"]) for sample in measured if sample.get("label") == label]
        )
    return {
        "tick_latency_seconds": summarize_values(tick_latencies),
        "first_tick_latency_seconds": summarize_values(first_tick_latencies),
        "steady_tick_latency_seconds": summarize_values(steady_tick_latencies),
        "realtime_factor": summarize_values(realtime_factors),
        "output_throughput_units_per_second": total_units / total_latency if total_latency else None,
        "peak_memory_mb": summarize_values(
            [float(sample["peak_memory_mb"]) for sample in measured if sample.get("peak_memory_mb") is not None]
        ),
        "stage_durations": {name: summarize_values(values) for name, values in sorted(stage_values.items())},
        "per_tick_index": per_tick_index,
        "per_label": per_label,
        "all_finite": all(bool(sample["finite"]) for sample in measured),
        "shapes": sorted({tuple(sample["shape"]) for sample in measured if sample.get("shape") is not None}),
    }


async def run_benchmark(
    adapter: InteractiveWorldModelAdapter,
    workload: WorkloadSpec,
    *,
    warmup_sessions: int,
    measured_sessions: int,
) -> dict[str, Any]:
    """Execute a deterministic workload while guaranteeing lifecycle cleanup."""

    if warmup_sessions < 0:
        raise ValueError("warmup_sessions must be non-negative")
    if measured_sessions <= 0:
        raise ValueError("measured_sessions must be positive")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "workload": workload.to_dict(),
        "warmup_sessions": warmup_sessions,
        "measured_sessions": measured_sessions,
        "samples": [],
    }
    shutdown_error: str | None = None
    try:
        initialize_started = perf_counter()
        runtime_metadata = await adapter.initialize(workload)
        result["initialize_seconds"] = perf_counter() - initialize_started
        if not isinstance(runtime_metadata, Mapping):
            raise BenchmarkContractError("adapter initialize() must return a metadata mapping")
        result["runtime_metadata"] = dict(runtime_metadata)

        samples: list[dict[str, Any]] = []
        result["samples"] = samples
        measured_create_latencies: list[float] = []
        measured_close_latencies: list[float] = []
        total_sessions = warmup_sessions + measured_sessions
        for session_index in range(total_sessions):
            measured = session_index >= warmup_sessions
            session_id = f"{workload.scenario_id}-{session_index}"
            created = False
            create_started = perf_counter()
            try:
                await adapter.create_session(session_id, workload)
                created = True
                create_latency = perf_counter() - create_started
                if measured:
                    measured_create_latencies.append(create_latency)
                for tick_index, tick in enumerate(workload.ticks):
                    tick_started = perf_counter()
                    response = await adapter.execute_tick(session_id, tick_index, tick)
                    latency_seconds = perf_counter() - tick_started
                    if not isinstance(response, Mapping):
                        raise BenchmarkContractError("adapter execute_tick() must return a mapping")
                    observation = _normalize_observation(
                        response,
                        session_id=session_id,
                        tick_index=tick_index,
                        tick=tick,
                    )
                    observation.update(
                        {
                            "session_index": session_index,
                            "measured": measured,
                            "latency_seconds": latency_seconds,
                            "label": tick.label,
                        }
                    )
                    samples.append(observation)
            finally:
                if created:
                    close_started = perf_counter()
                    await adapter.close_session(session_id)
                    close_latency = perf_counter() - close_started
                    if measured:
                        measured_close_latencies.append(close_latency)
        result["summary"] = summarize_samples(samples, output_rate_hz=workload.output_rate_hz)
        result["summary"]["session_create_seconds"] = summarize_values(measured_create_latencies)
        result["summary"]["session_close_seconds"] = summarize_values(measured_close_latencies)
        result["cold_first_tick_seconds"] = samples[0]["latency_seconds"] if samples else None
        result["status"] = "PASS"
    except BenchmarkContractError as exc:
        result.update(
            {
                "status": "INVALID_OUTPUT",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "RUNTIME_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        shutdown_started = perf_counter()
        try:
            await adapter.shutdown()
        except Exception as exc:
            shutdown_error = f"{type(exc).__name__}: {exc}"
            if result["status"] == "PASS":
                result.update({"status": "RUNTIME_ERROR", "error": shutdown_error})
        result["shutdown_seconds"] = perf_counter() - shutdown_started
        if shutdown_error is not None:
            result["shutdown_error"] = shutdown_error
    return result


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_markdown_summary(path: Path, result: Mapping[str, Any]) -> None:
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    tick = summary.get("tick_latency_seconds")
    steady = summary.get("steady_tick_latency_seconds")
    rtf = summary.get("realtime_factor")
    peak = summary.get("peak_memory_mb")

    def metric(value: Any, name: str) -> Any:
        return value.get(name) if isinstance(value, Mapping) else None

    rows = [
        ("Status", result.get("status")),
        ("Scenario", result.get("workload", {}).get("scenario_id")),
        ("Measured sessions", result.get("measured_sessions")),
        ("Tick latency P50 (s)", metric(tick, "p50")),
        ("Tick latency P95 (s)", metric(tick, "p95")),
        ("Steady latency P50 (s)", metric(steady, "p50")),
        ("RTF P50", metric(rtf, "p50")),
        ("Output throughput (units/s)", summary.get("output_throughput_units_per_second")),
        ("Peak memory max (MiB)", metric(peak, "max")),
    ]
    lines = [
        "# Interactive World Model Benchmark Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in rows:
        formatted = "" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {name} | {formatted} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an interactive world model through a generic adapter.")
    parser.add_argument("--adapter", required=True, help="Adapter factory as module.path:factory.")
    parser.add_argument("--adapter-args", default="{}", help="JSON object passed to the adapter factory.")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--warmup-sessions", type=int, default=1)
    parser.add_argument("--measured-sessions", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    adapter_args = json.loads(args.adapter_args)
    if not isinstance(adapter_args, Mapping):
        raise ValueError("--adapter-args must decode to a JSON object")
    workload = load_workload(args.workload)
    adapter = load_adapter(args.adapter, adapter_args)
    result = asyncio.run(
        run_benchmark(
            adapter,
            workload,
            warmup_sessions=args.warmup_sessions,
            measured_sessions=args.measured_sessions,
        )
    )
    output_dir = args.output_dir.expanduser().resolve()
    _atomic_write_json(output_dir / "result.json", result)
    write_markdown_summary(output_dir / "summary.md", result)
    print(json.dumps({"status": result["status"], "result": str(output_dir / "result.json")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
