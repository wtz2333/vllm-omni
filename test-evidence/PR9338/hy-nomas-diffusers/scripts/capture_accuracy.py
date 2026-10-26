import json
import os
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from skimage.metrics import structural_similarity


@pytest.fixture(autouse=True)
def capture_outputs(request, monkeypatch):
    if request.node.originalname != "test_reference_decoded_video":
        yield
        return
    reference, candidate, _ = request.getfixturevalue("pipelines")
    expected = actual = None
    reference_call = reference.inference
    candidate_call = candidate.forward

    def record_reference(*args, **kwargs):
        nonlocal expected
        result = reference_call(*args, **kwargs)
        expected = result.detach().cpu()
        return result

    def record_candidate(*args, **kwargs):
        nonlocal actual
        result = candidate_call(*args, **kwargs)
        actual = result.media.video.tensor.detach().cpu()
        return result

    monkeypatch.setattr(reference, "inference", record_reference)
    monkeypatch.setattr(candidate, "forward", record_candidate)
    yield
    assert expected is not None and actual is not None
    assert expected.shape == actual.shape == (1, 3, 49, 384, 640)
    scenario = request.node.callspec.params["scenario"]
    offload = request.node.callspec.params["offload"]
    name = f"{scenario}-{'offload' if offload else 'resident'}"
    folder = Path(os.environ["HY_ACCURACY_OUTPUT"]) / name
    folder.mkdir(parents=True)
    absolute_sum = squared_sum = maximum = 0.0
    ssim = []
    reference_frames, candidate_frames = [], []
    for index in range(49):
        left = expected[0, :, index].float().permute(1, 2, 0).numpy()
        right = actual[0, :, index].float().permute(1, 2, 0).numpy()
        difference = np.abs(left.astype(np.float64) - right)
        absolute_sum += difference.sum()
        squared_sum += np.square(difference).sum()
        maximum = max(maximum, float(difference.max()))
        ssim.append(float(structural_similarity(left, right, data_range=1.0, channel_axis=2)))
        reference_frames.append(np.rint(left.clip(0, 1) * 255).astype(np.uint8))
        candidate_frames.append(np.rint(right.clip(0, 1) * 255).astype(np.uint8))
    mse = squared_sum / expected.numel()
    metrics = {
        "scenario": scenario, "cpu_offload": offload, "frames": 49,
        "shape": list(expected.shape), "comparison_space": "decoded float pixels before uint8 conversion and MP4 encoding",
        "mae": absolute_sum / expected.numel(), "mse": mse, "max_abs": maximum,
        "psnr_db": "Infinity" if mse == 0 else float(-10 * np.log10(mse)),
        "mean_frame_ssim": float(np.mean(ssim)), "min_frame_ssim": min(ssim),
        "exact": maximum == 0.0,
    }
    assert metrics["exact"]
    for label, frames in (("reference", reference_frames), ("candidate", candidate_frames)):
        imageio.mimwrite(folder / f"{label}.mp4", frames, fps=24, codec="libx264", quality=8, macro_block_size=1)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(folder / "reference.mp4"), "-i", str(folder / "candidate.mp4"),
            "-filter_complex",
            "[0:v]drawtext=text=Reference:x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[r];"
            "[1:v]drawtext=text=Omni:x=12:y=12:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5[c];[r][c]hstack=inputs=2",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(folder / "comparison.mp4"),
        ], check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(folder / "comparison.mp4"),
            "-filter_complex", "fps=8,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            "-loop", "0", str(folder / "comparison.gif"),
        ], check=True,
    )
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
