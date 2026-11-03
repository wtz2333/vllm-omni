# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path

import pytest
import torch.multiprocessing as mp

from tests.diffusion.models.hy_nomas.test_ar import _window_pipeline_worker

pytestmark = [pytest.mark.local_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("world_size", [2, 4, 8, 16])
def test_window_decoder_pipeline_matches_replicated_stream(tmp_path, monkeypatch, world_size):
    from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder

    assert PixelUMMDCAELatentDecoder.window_error_sync_version == 1
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    result_path = tmp_path / "window-pipeline.json"
    mp.start_processes(
        _window_pipeline_worker,
        args=(world_size, str(tmp_path / "rendezvous"), str(result_path)),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    results = json.loads(result_path.read_text())
    assert len(results) == world_size
    assert all(result == results[0] for result in results)
    assert results[0]["committed"][-1] == 27
