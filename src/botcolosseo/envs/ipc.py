from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class WorkerTimeout(TimeoutError):
    pass


class RemoteWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Request:
    id: int
    command: str
    payload: object


@dataclass(frozen=True)
class _Response:
    id: int
    ok: bool
    payload: object
    error: str | None = None


def _serve_requests(
    handler: Callable[[str, object], object],
    requests: Any,
    responses: Any,
) -> None:
    while True:
        request = requests.get()
        if request.command == "__close__":
            try:
                handler("close", None)
                responses.put(_Response(request.id, True, None))
            except BaseException:
                responses.put(
                    _Response(request.id, False, None, traceback.format_exc())
                )
            return
        try:
            payload = handler(request.command, request.payload)
            responses.put(_Response(request.id, True, payload))
        except BaseException:
            responses.put(_Response(request.id, False, None, traceback.format_exc()))


class ProcessClient:
    def __init__(
        self,
        process: Any,
        requests: Any,
        responses: Any,
        *,
        timeout: float,
    ) -> None:
        self._process = process
        self._requests = requests
        self._responses = responses
        self._timeout = timeout
        self._next_id = 0
        self._pending: set[int] = set()
        self._closed = False

    @classmethod
    def start(
        cls,
        handler: Callable[[str, object], object],
        *,
        name: str,
        timeout: float,
    ) -> ProcessClient:
        if timeout <= 0:
            raise ValueError("Worker timeout must be positive")
        context = mp.get_context("spawn")
        requests = context.Queue(maxsize=1)
        responses = context.Queue(maxsize=1)
        process = context.Process(
            target=_serve_requests,
            args=(handler, requests, responses),
            name=name,
            daemon=True,
        )
        process.start()
        return cls(process, requests, responses, timeout=timeout)

    def submit(self, command: str, payload: object) -> int:
        if self._closed or not self._process.is_alive():
            raise RuntimeError("Worker is not running")
        request_id = self._next_id
        self._next_id += 1
        try:
            self._requests.put(
                _Request(request_id, command, payload), timeout=self._timeout
            )
        except queue.Full as exc:
            raise WorkerTimeout(f"Timed out submitting {command}") from exc
        self._pending.add(request_id)
        return request_id

    def receive(self, request_id: int) -> object:
        if request_id not in self._pending:
            raise ValueError(f"Unknown pending request: {request_id}")
        try:
            response = self._responses.get(timeout=self._timeout)
        except queue.Empty as exc:
            raise WorkerTimeout(f"Timed out awaiting request {request_id}") from exc
        if response.id != request_id:
            raise RuntimeError(f"Response ID mismatch: {response.id} != {request_id}")
        self._pending.remove(request_id)
        if not response.ok:
            raise RemoteWorkerError(response.error or "Remote worker failed")
        return response.payload

    def call(self, command: str, payload: object) -> object:
        return self.receive(self.submit(command, payload))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive() and not self._pending:
            request_id = self._next_id
            self._next_id += 1
            try:
                self._requests.put(
                    _Request(request_id, "__close__", None), timeout=self._timeout
                )
                response = self._responses.get(timeout=self._timeout)
                if response.id != request_id or not response.ok:
                    raise RemoteWorkerError(response.error or "Worker close failed")
            except (queue.Empty, queue.Full, RemoteWorkerError):
                pass
        self._process.join(timeout=self._timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=self._timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=self._timeout)
        self._requests.close()
        self._responses.close()

    def is_alive(self) -> bool:
        return self._process.is_alive()
