import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import psutil
import torch
from safetensors.torch import save_file

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def payloads(value):
    if isinstance(value, dict) and "pid" in value:
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in payloads(child)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in payloads(child)]
    return []


class HostSampler:
    def __init__(self):
        self.done = threading.Event()
        self.peak_rss = self.peak_pss = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.done.is_set():
            rss = pss = 0
            parent = psutil.Process()
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    memory = process.memory_full_info()
                    rss += memory.rss
                    pss += memory.pss
                except psutil.NoSuchProcess:
                    continue
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_pss = max(self.peak_pss, pss)
            self.done.wait(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / args.name
    output.mkdir(parents=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert head == "257856d3afce694803f83fd60674f292b7a34583"
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
    prompt = "A forest beside a clear river, photorealistic, daylight, gentle camera motion."
    manifest = {
        "head": head, "model": "tencent/Hy-Nomas-Diffusers",
        "model_revision": "a1ae04d11ae4ea176d1f76de634c115944f9882d",
        "sp": args.sp, "cpu_offload": args.offload, "prompt": prompt,
        "height": 704, "width": 1280, "frames": 121, "steps": 8, "seed": 42,
        "fps": 24, "guidance_scale": 1, "flow_shift": 5, "dtype": "bfloat16",
        "batch": 1, "concurrency": 1, "warmup": 2, "repeats": 5,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "timing": "Omni.generate call through complete CPU tensor delivery; excludes loading, metrics RPC, hashes, encoding and disk writes",
        "memory": "per-rank PyTorch peaks; host process-tree RSS/PSS sampled every second",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sampler = HostSampler()
    sampler.thread.start()
    engine = None
    try:
        started = time.perf_counter()
        engine = Omni(
            model=os.environ["HY_NOMAS_MODEL"], dtype="bfloat16", enforce_eager=True,
            distributed_executor_backend="mp", max_num_seqs=1,
            parallel_config={"ulysses_degree": args.sp}, enable_cpu_offload=args.offload,
            worker_extension_cls="hy_perf_worker.BenchmarkWorker",
        )
        manifest["load_ms"] = (time.perf_counter() - started) * 1000
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        hashes = []
        for index in range(7):
            before = payloads(engine.engine.collective_rpc("benchmark_memory", stage_ids=[0], kwargs={"reset": True}))
            assert len(before) == args.sp and all(x["sp_size"] == args.sp and x["cpu_offload"] == args.offload for x in before), before
            params = OmniDiffusionSamplingParams(
                height=704, width=1280, num_frames=121, num_inference_steps=8,
                seed=42, fps=24, guidance_scale=1, output_type="pt",
            )
            started = time.perf_counter()
            outputs = list(engine.generate({"prompt": prompt}, params))
            elapsed = (time.perf_counter() - started) * 1000
            after = payloads(engine.engine.collective_rpc("benchmark_memory", stage_ids=[0]))
            assert len(outputs) == 1 and len(after) == args.sp
            video = outputs[0].images
            if isinstance(video, list):
                video, = video
            assert isinstance(video, torch.Tensor) and tuple(video.shape) == (1, 121, 3, 704, 1280), getattr(video, "shape", None)
            assert video.device.type == "cpu" and torch.isfinite(video).all()
            video = video.permute(0, 2, 1, 3, 4).contiguous()
            digest = hashlib.sha256(video.view(torch.uint8).numpy()).hexdigest()
            hashes.append(digest)
            row = {
                "index": index, "warmup": index < 2, "success": True, "e2e_ms": elapsed,
                "stage_durations": outputs[0].stage_durations,
                "output_peak_memory_mb": outputs[0].peak_memory_mb,
                "ranks": after, "output_sha256": digest,
                "sampled_host_peak_rss_bytes": sampler.peak_rss, "sampled_host_peak_pss_bytes": sampler.peak_pss,
            }
            with (output / "runs.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if index == 2:
                save_file({"video": video}, str(output / "video.safetensors"))
            del video, outputs
        assert len(set(hashes)) == 1, "Repeated fixed-seed outputs differ"
    finally:
        if engine is not None:
            engine.close()
        sampler.done.set()
        sampler.thread.join()


if __name__ == "__main__":
    main()
