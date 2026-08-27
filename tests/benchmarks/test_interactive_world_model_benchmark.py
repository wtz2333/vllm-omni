# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchmarks.lingbot.interactive_world_model_benchmark import (
    TickSpec,
    WorkloadSpec,
    load_adapter,
    load_workload,
    run_benchmark,
    write_markdown_summary,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]

_EXAMPLE = Path(__file__).resolve().parents[2] / "benchmarks" / "lingbot" / "interactive_workload.example.json"


class FakeInteractiveAdapter:
    def __init__(
        self,
        *,
        corrupt_event_id: bool = False,
        fail_tick_index: int | None = None,
        stage_duration: object = 0.01,
    ) -> None:
        self.corrupt_event_id = corrupt_event_id
        self.fail_tick_index = fail_tick_index
        self.stage_duration = stage_duration
        self.initialize_calls = 0
        self.created_sessions: list[str] = []
        self.closed_sessions: list[str] = []
        self.active_sessions: set[str] = set()
        self.shutdown_calls = 0

    async def initialize(self, workload: WorkloadSpec) -> dict:
        self.initialize_calls += 1
        return {"runtime": "fake", "scenario_id": workload.scenario_id}

    async def create_session(self, session_id: str, workload: WorkloadSpec) -> None:
        assert workload.ticks
        self.created_sessions.append(session_id)
        self.active_sessions.add(session_id)

    async def execute_tick(self, session_id: str, tick_index: int, tick: TickSpec) -> dict:
        assert session_id in self.active_sessions
        if self.fail_tick_index == tick_index:
            raise RuntimeError("injected tick failure")
        return {
            "session_id": session_id,
            "event_id": tick.event_id + (1 if self.corrupt_event_id else 0),
            "tick_index": tick_index,
            "finite": True,
            "output_units": tick.expected_output_units or 1,
            "shape": [1, 4, 8, 8],
            "stage_durations": {"denoise": self.stage_duration, "decode": 0.002},
            "peak_memory_mb": 1024,
            "metadata": {"adapter": "fake"},
        }

    async def close_session(self, session_id: str) -> None:
        self.active_sessions.remove(session_id)
        self.closed_sessions.append(session_id)

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


def create_fake_adapter(adapter_args: dict) -> FakeInteractiveAdapter:
    return FakeInteractiveAdapter(corrupt_event_id=bool(adapter_args.get("corrupt_event_id", False)))


def _workload() -> WorkloadSpec:
    return WorkloadSpec.from_dict(
        {
            "scenario_id": "test-world",
            "output_rate_hz": 16,
            "metadata": {"seed": 42},
            "ticks": [
                {
                    "event_id": 0,
                    "prompt": "A fixed prompt.",
                    "controls": [{"track": "control", "data": {"value": 1}}],
                    "expected_output_units": 8,
                    "label": "first",
                },
                {
                    "event_id": 1,
                    "controls": [],
                    "expected_output_units": 8,
                    "label": "steady",
                },
                {
                    "event_id": 2,
                    "controls": [],
                    "expected_output_units": 8,
                    "label": "steady",
                },
            ],
        }
    )


def test_example_workload_preserves_opaque_controls() -> None:
    workload = load_workload(_EXAMPLE)

    assert workload.scenario_id == "generic_camera_navigation"
    assert workload.output_rate_hz == 16
    assert workload.ticks[0].controls[0]["schema"] == "example.direction.v1"
    assert workload.ticks[0].expected_output_units == 12


def test_dynamic_adapter_factory_uses_module_colon_factory_contract() -> None:
    adapter = load_adapter(
        f"{__name__}:create_fake_adapter",
        {"corrupt_event_id": True},
    )

    assert isinstance(adapter, FakeInteractiveAdapter)
    assert adapter.corrupt_event_id is True


def test_workload_rejects_non_monotonic_event_ids() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        WorkloadSpec.from_dict(
            {
                "scenario_id": "bad-events",
                "output_rate_hz": 16,
                "ticks": [
                    {"event_id": 1, "controls": []},
                    {"event_id": 1, "controls": []},
                ],
            }
        )


def test_benchmark_excludes_warmup_and_summarizes_tick_labels() -> None:
    adapter = FakeInteractiveAdapter()

    result = asyncio.run(
        run_benchmark(
            adapter,
            _workload(),
            warmup_sessions=1,
            measured_sessions=2,
        )
    )

    assert result["status"] == "PASS"
    assert adapter.initialize_calls == 1
    assert len(adapter.created_sessions) == 3
    assert adapter.closed_sessions == adapter.created_sessions
    assert adapter.active_sessions == set()
    assert adapter.shutdown_calls == 1
    assert len(result["samples"]) == 9
    assert result["summary"]["tick_latency_seconds"]["count"] == 6
    assert result["summary"]["first_tick_latency_seconds"]["count"] == 2
    assert result["summary"]["steady_tick_latency_seconds"]["count"] == 4
    assert result["summary"]["session_create_seconds"]["count"] == 2
    assert result["summary"]["session_close_seconds"]["count"] == 2
    assert result["summary"]["stage_durations"]["denoise"]["count"] == 6
    assert result["summary"]["per_label"]["first"]["count"] == 2
    assert result["summary"]["per_label"]["steady"]["count"] == 4
    assert result["summary"]["output_throughput_units_per_second"] > 0
    assert result["cold_first_tick_seconds"] is not None


def test_invalid_tick_correlation_fails_closed_and_cleans_up() -> None:
    adapter = FakeInteractiveAdapter(corrupt_event_id=True)

    result = asyncio.run(
        run_benchmark(
            adapter,
            _workload(),
            warmup_sessions=0,
            measured_sessions=1,
        )
    )

    assert result["status"] == "INVALID_OUTPUT"
    assert "event_id" in result["error"]
    assert adapter.created_sessions == ["test-world-0"]
    assert adapter.closed_sessions == ["test-world-0"]
    assert adapter.active_sessions == set()
    assert adapter.shutdown_calls == 1


def test_runtime_failure_preserves_partial_samples_and_cleans_up() -> None:
    adapter = FakeInteractiveAdapter(fail_tick_index=1)

    result = asyncio.run(
        run_benchmark(
            adapter,
            _workload(),
            warmup_sessions=0,
            measured_sessions=1,
        )
    )

    assert result["status"] == "RUNTIME_ERROR"
    assert "injected tick failure" in result["error"]
    assert len(result["samples"]) == 1
    assert adapter.closed_sessions == ["test-world-0"]
    assert adapter.shutdown_calls == 1


def test_invalid_stage_metric_is_classified_as_contract_failure() -> None:
    adapter = FakeInteractiveAdapter(stage_duration="not-a-number")

    result = asyncio.run(
        run_benchmark(
            adapter,
            _workload(),
            warmup_sessions=0,
            measured_sessions=1,
        )
    )

    assert result["status"] == "INVALID_OUTPUT"
    assert "stage durations must be numeric" in result["error"]
    assert adapter.closed_sessions == ["test-world-0"]
    assert adapter.shutdown_calls == 1


def test_markdown_summary_contains_decision_metrics(tmp_path: Path) -> None:
    result = asyncio.run(
        run_benchmark(
            FakeInteractiveAdapter(),
            _workload(),
            warmup_sessions=0,
            measured_sessions=1,
        )
    )
    output = tmp_path / "summary.md"

    write_markdown_summary(output, result)

    text = output.read_text()
    assert "Interactive World Model Benchmark Summary" in text
    assert "Tick latency P50" in text
    assert "Output throughput" in text
    assert "| Status | PASS |" in text
