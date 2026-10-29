# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import av
import httpx
import numpy as np
import torch
import yaml
from safetensors.torch import load_file
from websockets.asyncio.client import connect

REPO = Path(os.environ["VLLM_OMNI_SOURCE"]).resolve()
ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "A forest beside a clear river, photorealistic, daylight, gentle camera motion.",
    "Birds fly above the river as the camera approaches a wooden bridge, daylight.",
]
PORT = 18543


def inputs(args):
    if args.demo == "plain":
        return PROMPTS[0], {}
    return "|||".join(PROMPTS), {
        "segment_frame_boundaries": [0, 64, args.frames],
        "camera_trajectory": "pan_left",
        "segment_has_camera": [False, True],
    }


def config(args):
    value = yaml.safe_load((REPO / "vllm_omni/deploy/hy_nomas_ar_recent.yaml").read_text())
    stage = value["stages"][0]
    stage["devices"] = "0"
    stage["ulysses_degree"] = stage["sequence_parallel_size"] = 1
    stage["custom_pipeline_args"].update(
        ar_height=args.height,
        ar_width=args.width,
        ar_history_mode=args.history,
        ar_history_latent_frames=args.budget,
        ar_max_latent_frames=args.capacity,
    )
    return value


async def offline(args):
    from vllm_omni.diffusion.utils.media_utils import FragmentedMP4Muxer
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    os.environ["NOMAS_PARITY_CAPTURE_DIR"] = str(args.output / "offline")
    engine = AsyncOmni(
        model=os.environ["HY_NOMAS_MODEL"],
        enforce_eager=True,
        dtype="bfloat16",
        distributed_executor_backend="mp",
        max_num_seqs=1,
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        step_execution=False,
        worker_extension_cls="capture_worker.CaptureWorker",
        parallel_config={"ulysses_degree": 1},
        custom_pipeline_args=config(args)["stages"][0]["custom_pipeline_args"],
        model_config={"ar_diffusion_kv_config": {"gpu_memory_fraction": 0.02, "warmup_cudagraph": False}},
    )
    try:
        await engine.collective_rpc("qualification_setup", stage_ids=[0], timeout=180)
        prompt, extra = inputs(args)
        params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=4,
            seed=42,
            guidance_scale=1,
            fps=24,
            output_type="np",
            extra_args=extra,
        )
        final = None
        async for output in engine.generate(
            prompt, request_id="parity-offline", sampling_params_list=[params], output_modalities=["video"]
        ):
            final = output
        assert final is not None and final.finished and not final.error
        assert len(final.images) == 1
        video = final.images[0]
        assert isinstance(video, np.ndarray) and len(video) == args.frames and video.dtype == np.uint8
        np.save(args.output / "offline-output.npy", video)
        muxer = FragmentedMP4Muxer(width=args.width, height=args.height, fps=24)
        with (args.output / "offline.mp4").open("wb") as stream:
            stream.write(muxer.mux_video_frames(video))
            stream.write(muxer.close())
    finally:
        engine.shutdown()


async def websocket(args):
    prompt, extra = inputs(args)
    events = []
    pending = None
    async with connect(f"ws://127.0.0.1:{PORT}/v1/realtime/video", max_size=None, open_timeout=60) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.start",
                    "model": "Hy-Nomas",
                    "prompt": prompt,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.frames,
                    "num_inference_steps": 4,
                    "guidance_scale": 1,
                    "seed": 42,
                    "fps": 24,
                    "extra_params": extra,
                }
            )
        )

        async def heartbeat():
            while True:
                await asyncio.sleep(10)
                await ws.send(json.dumps({"type": "session.ping"}))

        ping = asyncio.create_task(heartbeat())
        try:
            with (args.output / "realtime.mp4").open("wb") as stream:
                async for message in ws:
                    if isinstance(message, bytes):
                        assert pending is not None and len(message) == pending["byte_length"]
                        stream.write(message)
                        pending = None
                        continue
                    event = json.loads(message)
                    events.append(event)
                    assert event["type"] != "error", event
                    if event["type"] == "video.chunk_metadata":
                        pending = event
                    if event["type"] == "session.done":
                        assert not event["stopped"]
                        break
        finally:
            ping.cancel()
            await asyncio.gather(ping, return_exceptions=True)
    (args.output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    assert events[-1]["type"] == "session.done"


def realtime(args):
    from tests.helpers.runtime import OmniServer

    deployment = args.output / "deploy.yaml"
    deployment.write_text(yaml.safe_dump(config(args), sort_keys=False))
    env = {**os.environ, "NOMAS_PARITY_CAPTURE_DIR": str(args.output / "realtime")}
    command = [
        sys.executable,
        str(Path(__file__).with_name("server.py")),
        "serve",
        os.environ["HY_NOMAS_MODEL"],
        "--omni",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--served-model-name",
        "Hy-Nomas",
        "--deploy-config",
        str(deployment),
        "--worker-extension-cls",
        "capture_worker.CaptureWorker",
        "--stage-init-timeout",
        "1800",
        "--init-timeout",
        "2100",
    ]
    owner = OmniServer(os.environ["HY_NOMAS_MODEL"], [], port=PORT)
    try:
        with (args.output / "server.log").open("w") as log:
            owner.proc = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + 2100
            while True:
                if owner.proc.poll() is not None:
                    raise RuntimeError(f"Server exited {owner.proc.returncode}")
                try:
                    response = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=3)
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup")
                time.sleep(1)
            response = httpx.post(
                f"http://127.0.0.1:{PORT}/qualification/control",
                json={"method": "qualification_setup"},
                timeout=180,
            )
            response.raise_for_status()
            asyncio.run(websocket(args))
    finally:
        owner.__exit__(None, None, None)


def compare(args):
    left = sorted((args.output / "offline").glob("chunk-*.safetensors"))
    right = sorted((args.output / "realtime").glob("chunk-*.safetensors"))
    assert left and len(left) == len(right)
    records = []
    for index, (first, second) in enumerate(zip(left, right)):
        expected, actual = load_file(str(first)), load_file(str(second))
        row = {"chunk": index}
        for key in ("latents", "video"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            row[key + "_max_abs"] = 0
        records.append(row)
    captured = np.concatenate([load_file(str(path))["video"].numpy() for path in left])
    np.testing.assert_array_equal(captured, np.load(args.output / "offline-output.npy"))
    for name in ("offline", "realtime"):
        with av.open(str(args.output / f"{name}.mp4")) as container:
            count = 0
            for count, frame in enumerate(container.decode(video=0), 1):
                assert (frame.width, frame.height) == (args.width, args.height)
                assert abs(float(frame.time) - (count - 1) / 24) < 1e-6
        assert count == args.frames
    result = {
        "history": args.history,
        "demo": args.demo,
        "frames": args.frames,
        "records": records,
        "scope": "exact pre-codec video and latents; encoded media frame count and PTS",
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("offline", "realtime", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", choices=("full", "recent"), required=True)
    parser.add_argument("--demo", choices=("plain", "segments"), default="plain")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--frames", type=int, default=129)
    parser.add_argument("--budget", type=int, default=18)
    parser.add_argument("--capacity", type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker == "offline":
        asyncio.run(offline(args))
    elif args.worker == "realtime":
        realtime(args)
    else:
        compare(args)
