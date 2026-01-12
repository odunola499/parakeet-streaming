from __future__ import annotations

import base64
import json
import logging
import functools
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import trio

from parakeet.config import Config
from parakeet.engine.asr_engine import ASREngine
from parakeet.engine.scheduler import StreamResult


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
        port: int = 8765,
        status_port: int | None = None,
        device: str = "cpu",
    ):
        self.config = config
        self.host = host
        self.port = port
        self.status_port = status_port
        self.device = device
        self.engine = ASREngine(config, device=device)
        self._conn_lock = trio.Lock()
        self._active_streams: set[int] = set()

    async def serve(self) -> None:
        logging.info("Started server on %s:%s", self.host, self.port)
        async with trio.open_nursery() as nursery:
            serve_main = functools.partial(
                trio.serve_tcp, self._handle_client, self.port, host=self.host
            )
            nursery.start_soon(serve_main)
            if self.status_port is not None:
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
            await stream.aclose()

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
        "is_final": result.is_final,
    }


async def _send_json(
    stream: trio.SocketStream, lock: trio.Lock, payload: dict[str, Any]
) -> None:
    message = json.dumps(payload, separators=(",", ":")) + "\n"
    data = message.encode("utf-8")
    async with lock:
        await stream.send_all(data)
