from __future__ import annotations

import sys

from botcolosseo.runtime import inspect_runtime


def main() -> int:
    report = inspect_runtime()
    print(report.to_json())
    python_ok = sys.version_info[:2] == (3, 10)
    vizdoom_ok = report.vizdoom_version == "1.3.0"
    return 0 if python_ok and vizdoom_ok else 1
