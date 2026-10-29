# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from fastapi import Request

from vllm_omni.entrypoints.cli.main import main
from vllm_omni.entrypoints.openai import api_server

original_build = api_server.build_openai_app


def build(*args, **kwargs):
    app = original_build(*args, **kwargs)

    @app.post("/qualification/control")
    async def control(request: Request):
        payload = await request.json()
        method = payload.pop("method")
        if method not in {"qualification_setup", "qualification_status", "qualification_fault"}:
            raise ValueError("Unsupported qualification method")
        return await app.state.engine_client.collective_rpc(method, kwargs=payload, stage_ids=[0], timeout=180)

    return app


api_server.build_openai_app = build

if __name__ == "__main__":
    main()
