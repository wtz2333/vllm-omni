# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]

_ROOT = Path(__file__).parents[2]
_WEB = _ROOT / "examples/online_serving/lingbot_world_realtime"


def test_lingbot_realtime_web_assets_expose_keyboard_and_pixel_transport() -> None:
    html = (_WEB / "index.html").read_text()
    javascript = (_WEB / "app.js").read_text()
    readme = (_WEB / "README.md").read_text()

    for key in "wasdijkl":
        assert f'data-key="{key}"' in html
    assert "/v1/realtime/world" in html
    assert 'type: "session.control"' in javascript
    assert 'socket.binaryType = "arraybuffer"' in javascript
    assert "createImageBitmap" in javascript
    assert "image_reference" in javascript
    assert "vllm_omni/deploy/lingbot_world_realtime.yaml" in readme


def test_lingbot_realtime_deploy_uses_ar_engine_sp2_and_spatial_vae2() -> None:
    deploy = yaml.safe_load((_ROOT / "vllm_omni/deploy/lingbot_world_realtime.yaml").read_text())
    stage = deploy["stages"][0]

    assert stage["engine_backend"].endswith(".ARDiffusionEngine")
    assert stage["enforce_eager"] is True
    assert stage["model_config"]["ar_diffusion_streaming_vae"] is True
    assert stage["parallel_config"]["sequence_parallel_size"] == 2
    assert stage["parallel_config"]["ulysses_degree"] == 2
    assert stage["parallel_config"]["vae_patch_parallel_size"] == 2
    assert stage["parallel_config"]["vae_parallel_mode"] == "spatial_shard_height"


def test_public_api_server_registers_realtime_world_route() -> None:
    source = (_ROOT / "vllm_omni/entrypoints/openai/api_server.py").read_text()
    assert '@router.websocket("/v1/realtime/world")' in source
    assert "OmniRealtimeWorldHandler" in source
