import json

from botcolosseo.runtime import RuntimeReport


def test_runtime_report_serializes_stable_keys() -> None:
    report = RuntimeReport(
        python_version="3.10.20",
        python_executable="/tmp/python",
        torch_version="2.6.0+cu124",
        vizdoom_version="1.3.0",
        cuda_available=False,
        cuda_version="12.4",
        gpu_names=(),
    )

    payload = json.loads(report.to_json())

    assert payload == {
        "cuda_available": False,
        "cuda_version": "12.4",
        "gpu_names": [],
        "python_executable": "/tmp/python",
        "python_version": "3.10.20",
        "torch_version": "2.6.0+cu124",
        "vizdoom_version": "1.3.0",
    }
