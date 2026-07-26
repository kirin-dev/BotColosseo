from __future__ import annotations

import socket

from botcolosseo.envs.synchronous_duel import allocate_loopback_port


def test_loopback_port_is_available_for_vizdoom_udp_host() -> None:
    port = allocate_loopback_port()

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", port))
