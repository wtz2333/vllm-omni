# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.entrypoints.openai.serving_realtime_world import (
    OmniRealtimeWorldHandler,
    RealtimeWorldSessionStart,
    _RealtimeWorldBackend,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionChunkMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _WebSocket:
    def __init__(self, messages: list[str]) -> None:
        self._messages: asyncio.Queue[str] = asyncio.Queue()
        for message in messages:
            self._messages.put_nowait(message)
        self.accepted = False
        self.closed = False
        self.sent_json: list[dict] = []
        self.sent_bytes: list[bytes] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        return await self._messages.get()

    async def send_json(self, value: dict) -> None:
        self.sent_json.append(value)

    async def send_bytes(self, value: bytes) -> None:
        self.sent_bytes.append(value)

    async def close(self, code: int = 1000) -> None:
        del code
        self.closed = True


class _Session:
    session_id = "world-test"

    def __init__(self) -> None:
        self.events = []

    async def accept_event(self, event):
        self.events.append(event)
        return SimpleNamespace(
            session_id=self.session_id,
            event_id=event.event_id,
            status=SimpleNamespace(value="accepted"),
            pending_event_count=1,
        )

    async def next_chunk(self):
        return SimpleNamespace(
            error=None,
            images=[torch.linspace(0, 1, 2 * 3 * 4 * 6).reshape(2, 3, 4, 6)],
        )


class _Consumer:
    @staticmethod
    def chunk_metadata(output) -> ARDiffusionChunkMetadata:
        del output
        return ARDiffusionChunkMetadata(
            session_id="world-test",
            request_id="world-test-0",
            chunk_index=0,
            applied_event_ids=(0,),
        )


class _Manager:
    def __init__(self) -> None:
        self.disconnected: list[str] = []

    async def disconnect(self, session_id: str) -> None:
        self.disconnected.append(session_id)


@pytest.mark.asyncio
async def test_public_world_websocket_streams_browser_decodable_pixel_frames() -> None:
    session = _Session()
    manager = _Manager()

    async def backend_factory(start, image):
        assert start.pixel_format == "jpeg"
        assert image.size == (6, 4)
        return _RealtimeWorldBackend(manager=manager, session=session, consumer=_Consumer())

    async def image_decoder(reference):
        assert reference == "test-image"
        return Image.new("RGB", (6, 4), color="navy")

    handler = OmniRealtimeWorldHandler(
        engine_client=SimpleNamespace(),
        backend_factory=backend_factory,
        image_decoder=image_decoder,
    )
    websocket = _WebSocket(
        [
            json.dumps(
                {
                    "type": "session.start",
                    "prompt": "a stable world",
                    "image_reference": "test-image",
                    "pixel_format": "jpeg",
                    "max_chunks": 1,
                }
            )
        ]
    )

    await handler.handle_session(websocket)  # type: ignore[arg-type]

    assert websocket.accepted
    assert [message["type"] for message in websocket.sent_json] == [
        "session.started",
        "video.chunk",
        "session.done",
    ]
    chunk = websocket.sent_json[1]
    assert chunk["chunk_index"] == 0
    assert chunk["frame_count"] == 2
    assert chunk["mime_type"] == "image/jpeg"
    assert chunk["byte_lengths"] == [len(value) for value in websocket.sent_bytes]
    assert len(websocket.sent_bytes) == 2
    assert all(value.startswith(b"\xff\xd8") and value.endswith(b"\xff\xd9") for value in websocket.sent_bytes)
    assert session.events[0].prompt == "a stable world"
    assert session.events[0].controls[0].data["mode"] == "state"
    assert manager.disconnected == ["world-test"]


def test_session_control_maps_held_keys_to_lingbot_state_transition() -> None:
    event = OmniRealtimeWorldHandler._control_event(
        {
            "type": "session.control",
            "event_id": 7,
            "actions": ["L", "w", "w"],
            "client_ts_ms": 1234,
            "prompt": "turn toward the bridge",
        }
    )

    assert event.event_id == 7
    assert event.prompt == "turn toward the bridge"
    assert len(event.controls) == 1
    assert event.controls[0].to_dict() == {
        "track": "camera",
        "schema": "lingbot.camera_actions.v1",
        "data": {
            "mode": "state",
            "transitions": [{"actions": ["w", "l"], "client_ts_ms": 1234}],
        },
    }


@pytest.mark.parametrize(
    "message",
    [
        {"type": "session.control", "event_id": 0, "actions": ["w"]},
        {"type": "session.control", "event_id": 1, "actions": ["x"]},
        {"type": "session.control", "event_id": 1},
        {"type": "session.control", "event_id": 1, "prompt": "  "},
    ],
)
def test_session_control_rejects_invalid_public_payloads(message: dict) -> None:
    with pytest.raises(ValueError):
        OmniRealtimeWorldHandler._control_event(message)


def test_session_start_normalizes_initial_actions_and_rejects_unknown_fields() -> None:
    start = RealtimeWorldSessionStart.model_validate(
        {
            "type": "session.start",
            "prompt": "world",
            "image_reference": "image",
            "initial_actions": ["D", "w", "d"],
        }
    )
    assert start.initial_actions == ["w", "d"]

    with pytest.raises(ValueError):
        RealtimeWorldSessionStart.model_validate(
            {
                "type": "session.start",
                "prompt": "world",
                "image_reference": "image",
                "unknown": True,
            }
        )
