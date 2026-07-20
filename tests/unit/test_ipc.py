import multiprocessing as mp
import time

import pytest

from botcolosseo.envs.ipc import ProcessClient, RemoteWorkerError, WorkerTimeout


def handle(command: str, payload: object) -> object:
    if command == "echo":
        return payload
    if command == "fail":
        raise RuntimeError("remote boom")
    if command == "hang":
        time.sleep(2)
        return None
    raise ValueError(command)


def test_process_client_supports_split_submit_receive_and_clean_close() -> None:
    client = ProcessClient.start(handle, name="test-echo", timeout=1.0)
    try:
        request_id = client.submit("echo", {"value": 7})
        assert client.receive(request_id) == {"value": 7}
        assert client.is_alive()
    finally:
        client.close()

    assert not client.is_alive()


def test_remote_error_preserves_message_and_worker_can_close() -> None:
    client = ProcessClient.start(handle, name="test-error", timeout=1.0)
    try:
        with pytest.raises(RemoteWorkerError, match="remote boom"):
            client.call("fail", None)
    finally:
        client.close()


def test_timeout_forces_bounded_worker_termination() -> None:
    before = {child.pid for child in mp.active_children()}
    client = ProcessClient.start(handle, name="test-hang", timeout=0.05)

    request_id = client.submit("hang", None)
    with pytest.raises(WorkerTimeout):
        client.receive(request_id)
    client.close()

    assert not client.is_alive()
    assert {child.pid for child in mp.active_children()} <= before
