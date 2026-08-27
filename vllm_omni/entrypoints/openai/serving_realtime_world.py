# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public WebSocket transport for interactive AR-Diffusion worlds.

The endpoint transports model-neutral session/control messages and browser-
decodable pixel frames. LingBot is the first adapter: held WASD/IJKL state is
reduced at AR chunk boundaries by :class:`LingBotCameraControlReducer`.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from vllm.logger import init_logger

from vllm_omni.diffusion.models.lingbot_world.actions import (
    LINGBOT_CAMERA_ACTION_SCHEMA,
    LingBotCameraControlReducer,
)
from vllm_omni.experimental.ar_diffusion.session import (
    ARDiffusionSession,
    ARDiffusionSessionError,
    ARDiffusionSessionEvent,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionControlInput,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
    from vllm_omni.experimental.ar_diffusion.session import ARDiffusionSessionManager
    from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_AR_ENGINE = "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
_VALID_ACTIONS = frozenset("wasdijkl")
_MAX_START_BYTES = 4 * 1024 * 1024
_MAX_CONTROL_BYTES = 128 * 1024
_START_TIMEOUT_SECONDS = 15.0
_CONTROL_POLL_SECONDS = 1.0


def _normalized_actions(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("actions must be a JSON array.")
    actions: set[str] = set()
    for item in value:
        if not isinstance(item, str) or item.lower() not in _VALID_ACTIONS:
            raise ValueError("actions supports only W/A/S/D/I/J/K/L keys.")
        actions.add(item.lower())
    return [key for key in "wasdijkl" if key in actions]


class RealtimeWorldSessionStart(BaseModel):
    """First client message for ``/v1/realtime/world``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["session.start"]
    model: str | None = None
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=16_384)
    image_reference: Any
    width: int = Field(default=832, ge=16, le=4096)
    height: int = Field(default=480, ge=16, le=4096)
    fps: int = Field(default=16, ge=1, le=120)
    seed: int = 42
    num_inference_steps: int = Field(default=4, ge=1, le=200)
    flow_shift: float = Field(default=5.0, gt=0)
    pixel_format: Literal["jpeg", "webp"] = "jpeg"
    pixel_quality: int = Field(default=85, ge=1, le=100)
    initial_actions: list[str] = Field(default_factory=list)
    max_chunks: int | None = Field(default=None, ge=1)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text.")
        return value

    @field_validator("initial_actions")
    @classmethod
    def _validate_initial_actions(cls, value: list[str]) -> list[str]:
        return _normalized_actions(value)


@dataclass
class _RealtimeWorldBackend:
    manager: ARDiffusionSessionManager[OmniRequestOutput]
    session: ARDiffusionSession[OmniRequestOutput]
    consumer: ARDiffusionOmniTickConsumer


BackendFactory = Callable[[RealtimeWorldSessionStart, Image.Image], Awaitable[_RealtimeWorldBackend]]
ImageDecoder = Callable[[object], Awaitable[Image.Image]]


class OmniRealtimeWorldHandler:
    """Serve one persistent interactive world per WebSocket connection."""

    def __init__(
        self,
        engine_client: AsyncOmni,
        model_name: str | None = None,
        stage_configs: list[Any] | None = None,
        *,
        backend_factory: BackendFactory | None = None,
        image_decoder: ImageDecoder | None = None,
    ) -> None:
        self._engine_client = engine_client
        self._model_name = model_name
        self._stage_configs = stage_configs
        self._backend_factory = backend_factory
        self._image_decoder = image_decoder or self._decode_image
        self._active_session_lock = asyncio.Lock()

    async def handle_session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await asyncio.wait_for(self._active_session_lock.acquire(), timeout=0.01)
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Another realtime world session is already active.", code="busy")
            await websocket.close(code=1013)
            return

        backend: _RealtimeWorldBackend | None = None
        control_task: asyncio.Task[None] | None = None
        stop_event = asyncio.Event()
        send_lock = asyncio.Lock()
        disconnected = False
        chunks_sent = 0
        try:
            start = await self._receive_start(websocket)
            if start is None:
                return
            if self._model_name is not None and start.model not in (None, self._model_name):
                await self._send_error(
                    websocket,
                    f"Model mismatch: server runs {self._model_name!r}, request uses {start.model!r}.",
                    code="model_mismatch",
                )
                return

            image = await self._image_decoder(start.image_reference)
            backend = await self._create_backend(start, image)
            await self._accept_initial_event(backend.session, start)
            session_id = backend.session.session_id
            mime_type = "image/jpeg" if start.pixel_format == "jpeg" else "image/webp"
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "session.started",
                        "session_id": session_id,
                        "pixel_format": start.pixel_format,
                        "mime_type": mime_type,
                        "width": start.width,
                        "height": start.height,
                        "fps": start.fps,
                        "first_chunk_frames": 9,
                        "later_chunk_frames": 12,
                        "next_event_id": 1,
                    }
                )

            control_task = asyncio.create_task(self._control_loop(websocket, backend.session, stop_event, send_lock))
            while not stop_event.is_set():
                if control_task.done():
                    disconnected = bool(control_task.result())
                    break
                started_at = time.perf_counter()
                output = await backend.session.next_chunk()
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                if output.error:
                    raise RuntimeError(str(output.error))
                metadata = backend.consumer.chunk_metadata(output)
                frames = self._extract_frames(output)
                encoded = await asyncio.to_thread(
                    self._encode_frames,
                    frames,
                    start.pixel_format,
                    start.pixel_quality,
                )
                async with send_lock:
                    await websocket.send_json(
                        {
                            "type": "video.chunk",
                            "session_id": session_id,
                            "request_id": metadata.request_id,
                            "chunk_index": metadata.chunk_index,
                            "applied_event_ids": list(metadata.applied_event_ids),
                            "frame_count": len(encoded),
                            "byte_lengths": [len(frame) for frame in encoded],
                            "mime_type": mime_type,
                            "latency_ms": latency_ms,
                        }
                    )
                    for frame in encoded:
                        await websocket.send_bytes(frame)
                chunks_sent += 1
                if start.max_chunks is not None and chunks_sent >= start.max_chunks:
                    stop_event.set()

            if not disconnected:
                async with send_lock:
                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "session_id": backend.session.session_id,
                            "chunks": chunks_sent,
                            "stopped": True,
                        }
                    )
        except WebSocketDisconnect:
            disconnected = True
        except HTTPException as exc:
            await self._send_error(websocket, str(exc.detail), code=exc.status_code)
        except ValidationError as exc:
            await self._send_error(websocket, f"Invalid request: {exc}", code="invalid_request")
        except Exception as exc:
            logger.exception("Realtime world WebSocket failed: %s", exc)
            await self._send_error(websocket, str(exc), code="internal_error")
        finally:
            stop_event.set()
            if control_task is not None:
                control_task.cancel()
                await asyncio.gather(control_task, return_exceptions=True)
            if backend is not None:
                try:
                    await backend.manager.disconnect(backend.session.session_id)
                except Exception:
                    logger.warning("Failed to close realtime world session", exc_info=True)
            self._active_session_lock.release()

    async def _receive_start(self, websocket: WebSocket) -> RealtimeWorldSessionStart | None:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=_START_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.start", code="start_timeout")
            return None
        if len(raw) > _MAX_START_BYTES:
            await self._send_error(websocket, "session.start message too large", code="message_too_large")
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.start", code="invalid_json")
            return None
        try:
            return RealtimeWorldSessionStart.model_validate(value)
        except ValidationError as exc:
            await self._send_error(websocket, f"Invalid session.start: {exc}", code="invalid_request")
            return None

    async def _create_backend(
        self,
        start: RealtimeWorldSessionStart,
        image: Image.Image,
    ) -> _RealtimeWorldBackend:
        if self._backend_factory is not None:
            return await self._backend_factory(start, image)
        from vllm_omni.entrypoints.openai.stage_params import (
            build_stage_sampling_params_list,
            get_default_sampling_params_list,
        )
        from vllm_omni.entrypoints.openai.utils import get_stage_type
        from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
        from vllm_omni.experimental.ar_diffusion.session import (
            ARDiffusionSessionManager,
            ARDiffusionWorkerLifecycle,
        )
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        stage_configs = self._stage_configs or self._engine_client.stage_configs
        if not stage_configs:
            raise HTTPException(503, "Stage configs are unavailable.")
        diffusion_stages = [
            (index, stage) for index, stage in enumerate(stage_configs) if get_stage_type(stage) == "diffusion"
        ]
        if len(diffusion_stages) != 1:
            raise HTTPException(503, "Realtime worlds require exactly one diffusion stage.")
        diffusion_stage_id, stage = diffusion_stages[0]
        if self._stage_engine_backend(stage) != _AR_ENGINE:
            raise HTTPException(503, "Realtime worlds require the ARDiffusionEngine backend.")

        sampling = OmniDiffusionSamplingParams(
            height=start.height,
            width=start.width,
            num_frames=9,
            num_inference_steps=start.num_inference_steps,
            max_sequence_length=512,
            seed=start.seed,
            fps=start.fps,
            output_type="pt",
            extra_args={"flow_shift": start.flow_shift},
        )
        sampling_params_list = build_stage_sampling_params_list(
            list(stage_configs),
            get_default_sampling_params_list(self._engine_client),
            diffusion_params=sampling,
            replace_diffusion_params=True,
        )
        consumer = ARDiffusionOmniTickConsumer(
            self._engine_client,
            prompt_provider=lambda tick: {
                "prompt": tick.prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params_list=sampling_params_list,
            diffusion_stage_id=diffusion_stage_id,
        )
        manager = ARDiffusionSessionManager(
            tick_consumer=consumer,
            lifecycle=ARDiffusionWorkerLifecycle(
                self._engine_client,
                stage_ids=[diffusion_stage_id],
                timeout=180.0,
            ),
            max_pending_events=32,
            control_reducer_factory=LingBotCameraControlReducer,
        )
        session = await manager.create_session(start.session_id or uuid.uuid4().hex)
        return _RealtimeWorldBackend(manager=manager, session=session, consumer=consumer)

    @staticmethod
    def _stage_engine_backend(stage: object) -> str | None:
        if isinstance(stage, Mapping):
            engine_args = stage.get("engine_args", stage)
        else:
            engine_args = getattr(stage, "engine_args", stage)
        if isinstance(engine_args, Mapping):
            value = engine_args.get("engine_backend")
        else:
            value = getattr(engine_args, "engine_backend", None)
        return value if isinstance(value, str) else None

    @staticmethod
    async def _decode_image(reference: object) -> Image.Image:
        from vllm_omni.entrypoints.openai.video_api_utils import decode_input_reference

        media = await decode_input_reference(reference, None, None)
        if not isinstance(media, Image.Image):
            raise HTTPException(400, "image_reference must resolve to exactly one image.")
        return media

    @staticmethod
    async def _accept_initial_event(
        session: ARDiffusionSession[OmniRequestOutput],
        start: RealtimeWorldSessionStart,
    ) -> None:
        control = ARDiffusionControlInput(
            track="camera",
            schema=LINGBOT_CAMERA_ACTION_SCHEMA,
            data={
                "mode": "state",
                "transitions": [{"actions": start.initial_actions}],
            },
        )
        await session.accept_event(
            ARDiffusionSessionEvent(
                event_id=0,
                prompt=start.prompt,
                controls=(control,),
            )
        )

    async def _control_loop(
        self,
        websocket: WebSocket,
        session: ARDiffusionSession[OmniRequestOutput],
        stop_event: asyncio.Event,
        send_lock: asyncio.Lock,
    ) -> bool:
        """Return true when the client disconnected."""

        while not stop_event.is_set():
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=_CONTROL_POLL_SECONDS)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                stop_event.set()
                return True
            if len(raw) > _MAX_CONTROL_BYTES:
                await self._send_error(websocket, "control message too large", code="message_too_large", lock=send_lock)
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error(websocket, "Invalid control JSON", code="invalid_json", lock=send_lock)
                continue
            if not isinstance(message, dict):
                await self._send_error(websocket, "Control message must be an object", lock=send_lock)
                continue
            msg_type = message.get("type")
            if msg_type == "session.stop":
                stop_event.set()
                return False
            if msg_type == "session.ping":
                async with send_lock:
                    await websocket.send_json({"type": "session.pong"})
                continue
            if msg_type != "session.control":
                await self._send_error(websocket, f"Unknown message type: {msg_type}", lock=send_lock)
                continue
            try:
                event = self._control_event(message)
                acceptance = await session.accept_event(event)
            except (ValueError, ARDiffusionSessionError) as exc:
                await self._send_error(websocket, str(exc), code="control_rejected", lock=send_lock)
                continue
            async with send_lock:
                await websocket.send_json(
                    {
                        "type": "session.control.queued",
                        "session_id": acceptance.session_id,
                        "event_id": acceptance.event_id,
                        "status": acceptance.status.value,
                        "pending_event_count": acceptance.pending_event_count,
                    }
                )
        return False

    @staticmethod
    def _control_event(message: Mapping[str, Any]) -> ARDiffusionSessionEvent:
        event_id = message.get("event_id")
        if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 1:
            raise ValueError("session.control event_id must be an integer >= 1.")
        prompt = message.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("session.control prompt must be non-empty when provided.")
        raw_actions = message.get("actions")
        controls: tuple[ARDiffusionControlInput, ...] = ()
        if raw_actions is not None:
            actions = _normalized_actions(raw_actions)
            timestamp = message.get("client_ts_ms")
            if timestamp is not None and (
                isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0
            ):
                raise ValueError("client_ts_ms must be a non-negative integer.")
            transition: dict[str, Any] = {"actions": actions}
            if timestamp is not None:
                transition["client_ts_ms"] = timestamp
            controls = (
                ARDiffusionControlInput(
                    track="camera",
                    schema=LINGBOT_CAMERA_ACTION_SCHEMA,
                    data={"mode": "state", "transitions": [transition]},
                ),
            )
        return ARDiffusionSessionEvent(event_id=event_id, prompt=prompt, controls=controls)

    @staticmethod
    def _extract_frames(output: OmniRequestOutput) -> list[Image.Image]:
        if not output.images or len(output.images) != 1:
            raise RuntimeError("Realtime world output must contain one pixel video tensor.")
        video = output.images[0]
        if isinstance(video, list) and all(isinstance(frame, Image.Image) for frame in video):
            return [frame.convert("RGB") for frame in video]
        if isinstance(video, np.ndarray):
            tensor = torch.from_numpy(video)
        elif isinstance(video, torch.Tensor):
            tensor = video.detach().cpu()
        else:
            raise RuntimeError(f"Unsupported realtime pixel payload: {type(video).__name__}.")
        if tensor.ndim == 5 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise RuntimeError(f"Realtime pixel tensor must be rank 4, got {tuple(tensor.shape)}.")
        if tensor.shape[1] in (1, 3, 4):
            tensor = tensor.permute(0, 2, 3, 1)
        elif tensor.shape[-1] not in (1, 3, 4):
            raise RuntimeError(f"Realtime pixel tensor has no image channel dimension: {tuple(tensor.shape)}.")
        if tensor.dtype == torch.uint8:
            pixels = tensor.numpy()
        else:
            tensor = tensor.float()
            if tensor.numel() and tensor.min() < 0:
                tensor = tensor / 2 + 0.5
            elif tensor.numel() and tensor.max() > 1:
                tensor = tensor / 255
            pixels = tensor.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        return [Image.fromarray(frame).convert("RGB") for frame in pixels]

    @staticmethod
    def _encode_frames(frames: list[Image.Image], pixel_format: str, quality: int) -> list[bytes]:
        format_name = "JPEG" if pixel_format == "jpeg" else "WEBP"
        encoded: list[bytes] = []
        for frame in frames:
            buffer = io.BytesIO()
            frame.save(buffer, format=format_name, quality=quality)
            encoded.append(buffer.getvalue())
        return encoded

    @staticmethod
    async def _send_error(
        websocket: WebSocket,
        message: str,
        *,
        code: str | int | None = None,
        lock: asyncio.Lock | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "error", "message": message}
        if code is not None:
            payload["code"] = code
        try:
            if lock is None:
                await websocket.send_json(payload)
            else:
                async with lock:
                    await websocket.send_json(payload)
        except Exception:
            pass
