# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Interactions that land between two steps wait for the next step instead of failing."""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CAMERA: dict[str, Any] = {
    "event_id": "ev-1",
    "event": {
        "multi_modal_data": {
            "camera": {"mode": "velocity", "data": {"translation": [0, 0, 0.05], "rotation": [0, 0, 0, 1]}}
        }
    },
    "received_at": 12.5,
}


class FakeCoordinator:
    """Records what the runner hands to the pipeline's interaction coordinator."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, list]] = []

    def enqueue_parts(self, state, *, parts, event_id, received_at, transition_chunks) -> None:
        self.calls.append((state.request_id, event_id, received_at, parts))


def make_runner() -> DiffusionModelRunner:
    runner = DiffusionModelRunner.__new__(DiffusionModelRunner)
    runner.pipeline = object()
    runner._interaction_coordinator = FakeCoordinator()
    runner.od_config = SimpleNamespace(streaming_output=True)
    runner.state_cache = {}
    runner._pending_interactions = {}
    runner._supports_step_mode = lambda: True  # type: ignore[method-assign]
    return runner


def test_an_interaction_with_no_live_step_is_queued_rather_than_rejected():
    runner = make_runner()

    runner.submit_interaction("req-1", CAMERA)

    assert runner._pending_interactions == {"req-1": [CAMERA]}
    assert runner._interaction_coordinator.calls == []


def test_the_queued_interaction_is_applied_on_the_next_step():
    runner = make_runner()
    runner.submit_interaction("req-1", CAMERA)

    runner._flush_pending_interactions(SimpleNamespace(request_id="req-1"))

    (call,) = runner._interaction_coordinator.calls
    request_id, event_id, received_at, parts = call
    assert (request_id, event_id, received_at) == ("req-1", "ev-1", 12.5)
    assert parts == [("camera", CAMERA["event"]["multi_modal_data"]["camera"])]
    assert runner._pending_interactions == {}


def test_an_interaction_with_a_live_step_is_delivered_at_once():
    runner = make_runner()
    runner.state_cache["req-1"] = SimpleNamespace(request_id="req-1")

    runner.submit_interaction("req-1", CAMERA)

    assert len(runner._interaction_coordinator.calls) == 1
    assert runner._pending_interactions == {}


def test_the_queue_is_bounded_and_drops_the_oldest_event():
    runner = make_runner()
    for index in range(40):
        runner.submit_interaction("req-1", {**CAMERA, "event_id": f"ev-{index}"})

    queued = runner._pending_interactions["req-1"]
    assert len(queued) == 32
    assert queued[0]["event_id"] == "ev-8"
    assert queued[-1]["event_id"] == "ev-39"


def test_a_retired_request_takes_its_queue_with_it():
    runner = make_runner()
    runner.submit_interaction("req-1", CAMERA)

    runner.release_step_requests(["req-1"])

    assert runner._pending_interactions == {}
