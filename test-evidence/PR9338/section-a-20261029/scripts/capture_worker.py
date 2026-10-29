# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


class CaptureWorker:
    def qualification_setup(self):
        directory = Path(os.environ["NOMAS_PARITY_CAPTURE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        self.parity_rows = []
        pipe = self.model_runner.pipeline
        original = pipe.post_decode

        def capture(state, **kwargs):
            latent = state.latents.detach().cpu().contiguous()
            output = original(state, **kwargs)
            video = np.ascontiguousarray(output.output["payload"]["video"])
            index = len(self.parity_rows)
            save_file(
                {"latents": latent, "video": torch.from_numpy(video)}, str(directory / f"chunk-{index:04d}.safetensors")
            )
            self.parity_rows.append(output.output["metadata"]["hy_nomas"])
            (directory / "rows.json").write_text(json.dumps(self.parity_rows, indent=2) + "\n")
            return output

        pipe.post_decode = capture
        return {"ready": True, "directory": str(directory)}

    def qualification_status(self):
        return {"chunks": len(self.parity_rows)}
