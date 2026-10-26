import os

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.platforms import current_omni_platform


class BenchmarkWorker:
    def benchmark_memory(self, reset=False):
        from hymm.distributed.parallel_states import get_sequence_parallel_state, nccl_info

        if reset:
            current_omni_platform.synchronize()
            current_omni_platform.reset_peak_memory_stats()
        pipeline = self.model_runner.pipeline
        row = {
            "pid": os.getpid(),
            "sp_size": pipeline.sp_size,
            "cpu_offload": pipeline._cpu_offload,
            "allocated_bytes": torch.accelerator.memory_allocated(),
            "reserved_bytes": current_omni_platform.memory_reserved(),
            "peak_allocated_bytes": current_omni_platform.max_memory_allocated(),
            "peak_reserved_bytes": current_omni_platform.max_memory_reserved(),
            "reference_sp_size": nccl_info.sp_size,
            "reference_sp_enabled": get_sequence_parallel_state(),
        }
        if pipeline.sp_size == 1:
            return [row]
        assert row["reference_sp_size"] == pipeline.sp_size and row["reference_sp_enabled"]
        rows = [None] * pipeline.sp_size
        dist.all_gather_object(rows, row, group=get_sp_group().cpu_group)
        return rows
