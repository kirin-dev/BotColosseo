from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass

import torch
import vizdoom as vzd


@dataclass(frozen=True)
class RuntimeReport:
    python_version: str
    python_executable: str
    torch_version: str
    vizdoom_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_names: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def inspect_runtime() -> RuntimeReport:
    cuda_available = torch.cuda.is_available()
    gpu_names = (
        tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
        if cuda_available
        else ()
    )
    return RuntimeReport(
        python_version=platform.python_version(),
        python_executable=sys.executable,
        torch_version=torch.__version__,
        vizdoom_version=vzd.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        gpu_names=gpu_names,
    )
