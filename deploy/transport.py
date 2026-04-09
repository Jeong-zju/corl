from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import zmq


def require_zmq() -> Any:
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "ZeroMQ transport requires `pyzmq`. Install it in the runtime environment first."
        ) from exc
    return zmq


def make_socket(
    ctx: "zmq.Context",
    socket_type: int,
    endpoint: str,
    *,
    bind: bool,
    linger_ms: int = 0,
    snd_hwm: int = 10,
    rcv_hwm: int = 10,
) -> "zmq.Socket":
    zmq = require_zmq()
    socket = ctx.socket(socket_type)
    socket.setsockopt(zmq.LINGER, int(linger_ms))
    socket.setsockopt(zmq.SNDHWM, int(snd_hwm))
    socket.setsockopt(zmq.RCVHWM, int(rcv_hwm))
    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)
    return socket


def close_socket(socket: "zmq.Socket") -> None:
    socket.setsockopt(require_zmq().LINGER, 0)
    socket.close()

