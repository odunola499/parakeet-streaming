import base64
import functools
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import trio

from parakeet.config import Config
from parakeet.engine.asr_engine import ASREngine
from parakeet.engine.scheduler import StreamResult
from trio_websocket import ConnectionClosed, serve_websocket


@dataclass
class ConnectionState:
    stream_id: int
    final_requested: bool = False
    disconnected: bool = False


class ASRSocketServer:
    def __init__(
        self,
        config: Config,
        host: str = "0.0.0.0",
        tcp_port: int | None = None,
        ws_port: int = 9000,
        status_port: int = 8000,
        device: str = "cuda",
    ):
        self.config = config
        self.host = host
        self.tcp_port = tcp_port
        self.ws_port = ws_port
        self.status_port = status_port
        self.device = device
        self.engine = ASREngine(config, device=device)
        self._conn_lock = trio.Lock()
        self._active_streams: set[int] = set()

    async def serve(self) -> None:
        logging.info("Started TCP server on %s:%s", self.host, self.tcp_port)
        async with trio.open_nursery() as nursery:

            # TCP
            if self.tcp_port is not None:
                logging.info("Started TCP server on %s:%s", self.host, self.tcp_port)
                serve_tcp = functools.partial(
                    trio.serve_tcp, self._handle_client, self.tcp_port, host=self.host
                )
                nursery.start_soon(serve_tcp)

            # WS
            logging.info("Started WebSocket server on %s:%s", self.host, self.ws_port)
            serve_ws = functools.partial(
                serve_websocket,
                self._handle_ws_client,
                self.host,
                self.ws_port,
                ssl_context=None,
            )
            nursery.start_soon(serve_ws)

            # Metrics
            logging.info(
                "Starting status endpoint on %s:%s", self.host, self.status_port
            )
            serve_status = functools.partial(
                trio.serve_tcp,
                self._handle_status,
                self.status_port,
                host=self.host,
            )
            nursery.start_soon(serve_status)

    def close(self) -> None:
        self.engine.close()

    async def _handle_client(self, stream: trio.SocketStream) -> None:
        peer = _format_peer(stream)
        try:
            stream_id = self.engine.create_stream()
        except RuntimeError as exc:
            send_lock = trio.Lock()
            await _send_json(
                stream,
                send_lock,
                {"type": "error", "message": str(exc)},
            )
            await stream.aclose()
            return

        state = ConnectionState(stream_id=stream_id)
        await self._register_stream(state.stream_id)
        logging.info("TCP client connected stream_id=%s peer=%s", stream_id, peer)
        send_lock = trio.Lock()
        await _send_json(
            stream,
            send_lock,
            {
                "type": "hello",
                "stream_id": state.stream_id,
                "sample_rate": self.config.sample_rate,
            },
        )

        try:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(
                    self._reader_loop, stream, state, send_lock, nursery.cancel_scope
                )
                nursery.start_soon(
                    self._writer_loop, stream, state, send_lock, nursery.cancel_scope
                )
        finally:
            await self._finalize_stream(state)
            await self._unregister_stream(state.stream_id)
            logging.info(
                "TCP client disconnected stream_id=%s peer=%s",
                state.stream_id,
                peer,
            )
            await stream.aclose()

    async def _handle_ws_client(self, request: Any) -> None:
        ws = await request.accept()
        peer = _format_ws_peer(request)
        try:
            stream_id = self.engine.create_stream()
        except RuntimeError as exc:
            send_lock = trio.Lock()
            await _send_ws_json(
                ws,
                send_lock,
                {"type": "error", "message": str(exc)},
            )
            await _close_ws(ws)
            return

        state = ConnectionState(stream_id=stream_id)
        await self._register_stream(state.stream_id)
        logging.info("WS client connected stream_id=%s peer=%s", stream_id, peer)
        send_lock = trio.Lock()
        await _send_ws_json(
            ws,
            send_lock,
            {
                "type": "hello",
                "stream_id": state.stream_id,
                "sample_rate": self.config.sample_rate,
            },
        )

        try:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(
                    self._reader_loop_ws, ws, state, send_lock, nursery.cancel_scope
                )
                nursery.start_soon(
                    self._writer_loop_ws, ws, state, send_lock, nursery.cancel_scope
                )
        finally:
            await self._finalize_stream(state)
            await self._unregister_stream(state.stream_id)
            logging.info(
                "WS client disconnected stream_id=%s peer=%s",
                state.stream_id,
                peer,
            )
            await _close_ws(ws)

    async def _reader_loop(
        self,
        stream: trio.SocketStream,
        state: ConnectionState,
        send_lock: trio.Lock,
        cancel_scope: trio.CancelScope,
    ) -> None:
        buffer = bytearray()
        try:
            while True:
                data = await stream.receive_some(4096)
                if not data:
                    state.disconnected = True
                    cancel_scope.cancel()
                    return
                buffer.extend(data)
                for message in _drain_messages(buffer):
                    msg_type = message.get("type")
                    if msg_type == "audio":
                        samples, final = _parse_audio_message(message)
                        if samples.size or final:
                            self.engine.push_samples(
                                state.stream_id, samples, final=final
                            )
                        if final:
                            state.final_requested = True
                            return
                    elif msg_type == "close":
                        self._request_final(state)
                        return
                    elif msg_type == "ping":
                        await _send_json(stream, send_lock, {"type": "pong"})
                    else:
                        await _send_json(
                            stream,
                            send_lock,
                            {"type": "error", "message": "unknown message type"},
                        )
        except trio.BrokenResourceError:
            state.disconnected = True
            cancel_scope.cancel()
        except Exception:
            logging.exception("Reader loop failed for stream %s", state.stream_id)
            state.disconnected = True
            cancel_scope.cancel()

    async def _reader_loop_ws(
        self,
        ws: Any,
        state: ConnectionState,
        send_lock: trio.Lock,
        cancel_scope: trio.CancelScope,
    ) -> None:
        try:
            while True:
                message = await _ws_get_message(ws)
                if message is None:
                    state.disconnected = True
                    cancel_scope.cancel()
                    return
                for payload in _drain_ws_messages(message):
                    msg_type = payload.get("type")
                    if msg_type == "audio":
                        samples, final = _parse_audio_message(payload)
                        if samples.size or final:
                            self.engine.push_samples(
                                state.stream_id, samples, final=final
                            )
                        if final:
                            state.final_requested = True
                            return
                    elif msg_type == "close":
                        self._request_final(state)
                        return
                    elif msg_type == "ping":
                        await _send_ws_json(ws, send_lock, {"type": "pong"})
                    else:
                        await _send_ws_json(
                            ws,
                            send_lock,
                            {"type": "error", "message": "unknown message type"},
                        )
        except ConnectionClosed:
            state.disconnected = True
            cancel_scope.cancel()
        except Exception:
            logging.exception("WS reader loop failed for stream %s", state.stream_id)
            state.disconnected = True
            cancel_scope.cancel()

    async def _writer_loop(
        self,
        stream: trio.SocketStream,
        state: ConnectionState,
        send_lock: trio.Lock,
        cancel_scope: trio.CancelScope,
    ) -> None:
        try:
            last_seq = self.engine.get_update_seq()
            while True:
                results = self.engine.collect_stream_results(state.stream_id)
                for result in results:
                    payload = _result_payload(result)
                    await _send_json(stream, send_lock, payload)
                    if result.is_final:
                        state.final_requested = True
                        cancel_scope.cancel()
                        return
                if results:
                    continue
                last_seq = await trio.to_thread.run_sync(
                    self.engine.wait_for_update,
                    last_seq,
                    abandon_on_cancel=True,
                )
        except trio.BrokenResourceError:
            state.disconnected = True
            cancel_scope.cancel()
        except Exception:
            logging.exception("Writer loop failed for stream %s", state.stream_id)
            state.disconnected = True
            cancel_scope.cancel()

    async def _writer_loop_ws(
        self,
        ws: Any,
        state: ConnectionState,
        send_lock: trio.Lock,
        cancel_scope: trio.CancelScope,
    ) -> None:
        try:
            last_seq = self.engine.get_update_seq()
            while True:
                results = self.engine.collect_stream_results(state.stream_id)
                for result in results:
                    payload = _result_payload(result)
                    await _send_ws_json(ws, send_lock, payload)
                    if result.is_final:
                        state.final_requested = True
                        cancel_scope.cancel()
                        return
                if results:
                    continue
                last_seq = await trio.to_thread.run_sync(
                    self.engine.wait_for_update,
                    last_seq,
                    abandon_on_cancel=True,
                )
        except ConnectionClosed:
            state.disconnected = True
            cancel_scope.cancel()
        except Exception:
            logging.exception("WS writer loop failed for stream %s", state.stream_id)
            state.disconnected = True
            cancel_scope.cancel()

    def _request_final(self, state: ConnectionState) -> None:
        if state.final_requested:
            return
        state.final_requested = True
        try:
            self.engine.push_samples(
                state.stream_id, np.empty((0,), dtype=np.float32), final=True
            )
        except KeyError:
            return

    async def _finalize_stream(self, state: ConnectionState) -> None:
        if self.engine.get_stream(state.stream_id) is None:
            self.engine.drop_stream_results(state.stream_id)
            return
        if not state.final_requested:
            self._request_final(state)
        last_seq = self.engine.get_update_seq()
        deadline = None
        if state.disconnected:
            deadline = trio.current_time() + 5.0
        while True:
            seq = self.engine.get_stream(state.stream_id)
            if seq is None:
                self.engine.drop_stream_results(state.stream_id)
                return
            with seq.lock:
                finished = seq.is_finished
            if finished:
                self.engine.cleanup_stream(state.stream_id)
                self.engine.drop_stream_results(state.stream_id)
                return
            self.engine.collect_stream_results(state.stream_id)
            if deadline is not None and trio.current_time() >= deadline:
                logging.warning(
                    "Forcing stream cleanup after disconnect stream_id=%s",
                    state.stream_id,
                )
                self.engine.cleanup_stream(state.stream_id)
                self.engine.drop_stream_results(state.stream_id)
                return
            last_seq = await trio.to_thread.run_sync(
                self.engine.wait_for_update,
                last_seq,
                abandon_on_cancel=True,
            )

    async def _handle_status(self, stream: trio.SocketStream) -> None:
        payload = await self._status_payload()
        message = json.dumps(payload, separators=(",", ":")) + "\n"
        await stream.send_all(message.encode("utf-8"))
        await stream.aclose()

    async def _status_payload(self) -> dict[str, Any]:
        async with self._conn_lock:
            active = len(self._active_streams)
        return {"type": "status", "connected_streams": active}

    async def _register_stream(self, stream_id: int) -> None:
        async with self._conn_lock:
            self._active_streams.add(stream_id)

    async def _unregister_stream(self, stream_id: int) -> None:
        async with self._conn_lock:
            self._active_streams.discard(stream_id)


def _drain_messages(buffer: bytearray) -> Iterable[dict[str, Any]]:
    while True:
        newline = buffer.find(b"\n")
        if newline == -1:
            break
        raw_line = buffer[:newline]
        del buffer[: newline + 1]
        if not raw_line.strip():
            continue
        try:
            yield json.loads(raw_line.decode("utf-8"))
        except json.JSONDecodeError:
            continue


def _drain_ws_messages(message: str | None) -> Iterable[dict[str, Any]]:
    if not message:
        return
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return
    for line in message.splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _parse_audio_message(message: dict[str, Any]) -> tuple[np.ndarray, bool]:
    final = bool(message.get("final", False))
    if "samples" in message:
        samples = np.asarray(message["samples"], dtype=np.float32)
        return samples, final
    data = message.get("data")
    if not data:
        return np.empty((0,), dtype=np.float32), final
    raw = base64.b64decode(data)
    encoding = message.get("encoding", "pcm16")
    if encoding == "pcm16":
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, final
    if encoding == "f32":
        samples = np.frombuffer(raw, dtype=np.float32)
        return samples, final
    raise ValueError(f"Unsupported encoding: {encoding}")


def _result_payload(result: StreamResult) -> dict[str, Any]:
    return {
        "type": "result",
        "stream_id": result.stream_id,
        "text": result.text,
        "token_ids": result.token_ids,
        "confidence_scores": result.confidence_scores,
        "is_final": result.is_final,
        "last_state": result.last_state,
        "turn_detection": result.turn_detection,
    }


async def _send_json(
    stream: trio.SocketStream, lock: trio.Lock, payload: dict[str, Any]
) -> None:
    message = json.dumps(payload, separators=(",", ":")) + "\n"
    data = message.encode("utf-8")
    async with lock:
        await stream.send_all(data)


def _format_peer(stream: trio.SocketStream) -> str:
    try:
        peer = stream.socket.getpeername()
    except Exception:
        return "unknown"
    return str(peer)


def _format_ws_peer(request: Any) -> str:
    for attr in ("remote", "client", "peer"):
        value = getattr(request, attr, None)
        if value:
            return str(value)
    return "unknown"


async def _send_ws_json(ws: Any, lock: trio.Lock, payload: dict[str, Any]) -> None:
    message = json.dumps(payload, separators=(",", ":"))
    async with lock:
        await _ws_send_message(ws, message)


async def _close_ws(ws: Any) -> None:
    close = getattr(ws, "aclose", None)
    if close is None:
        close = getattr(ws, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def _ws_get_message(ws: Any) -> Any:
    getter = getattr(ws, "get_message", None)
    if getter is None:
        getter = getattr(ws, "receive_message", None)
    if getter is None:
        raise RuntimeError("WebSocket connection has no message receiver.")
    result = getter()
    if inspect.isawaitable(result):
        return await result
    return result


async def _ws_send_message(ws: Any, message: str) -> None:
    sender = getattr(ws, "send_message", None)
    if sender is None:
        sender = getattr(ws, "send", None)
    if sender is None:
        raise RuntimeError("WebSocket connection has no message sender.")
    result = sender(message)
    if inspect.isawaitable(result):
        await result
