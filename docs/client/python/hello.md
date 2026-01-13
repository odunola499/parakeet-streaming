# Python client: hello and protocol guide

This guide explains how to connect to the streaming server, verify the connection, and
exchange messages reliably from Python. It covers both the raw TCP protocol and the
native WebSocket protocol.

## Before you start

The server must be running before you connect. The simplest verification is to
connect and read the initial `hello` message, then send a `ping` and expect `pong`.

WebSocket verification:
```python
import asyncio
import json
import websockets

async def verify_ws(uri: str) -> None:
    async with websockets.connect(uri) as ws:
        hello = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "ping"}))
        pong = json.loads(await ws.recv())
        print("hello:", hello)
        print("pong:", pong)

asyncio.run(verify_ws("ws://127.0.0.1:8766"))
```

TCP verification:
```python
import json
import socket

sock = socket.create_connection(("127.0.0.1", 8765))
buffer = b""

def read_line():
    global buffer
    while b"\n" not in buffer:
        data = sock.recv(4096)
        if not data:
            return None
        buffer += data
    line, buffer = buffer.split(b"\n", 1)
    return json.loads(line.decode("utf-8"))

hello = read_line()
sock.sendall(b"{\"type\":\"ping\"}\\n")
pong = read_line()
print("hello:", hello)
print("pong:", pong)
sock.close()
```

## Connection options

The server listens on two different ports when both are enabled:

Raw TCP (`--port`) uses newline-delimited JSON. Each JSON object is terminated with a
single `\n`.

WebSocket (`--ws-port`) uses standard WebSocket frames. Each message is a complete JSON
object sent as a text frame.

If you want browser compatibility or standard WebSocket tooling, use WebSockets. If you
are building a backend service and want the simplest transport, TCP is fine.

## Handshake and verification

On connect, the server sends a `hello` message with a `stream_id` and `sample_rate`. A
simple way to verify the connection is to read the `hello` message and then send a
`ping` and check for a `pong`.

Example `hello` message:
```json
{"type":"hello","stream_id":123,"sample_rate":16000}
```

Example `ping` / `pong` exchange:
```json
{"type":"ping"}
```
```json
{"type":"pong"}
```

## Message types

The server understands a small set of message types. The table below shows the common
flow for a client connection. `status` is returned only by the status port.

| Direction | type   | Purpose |
| --- | --- | --- |
| client -> server | audio  | Send audio samples for decoding |
| client -> server | close  | Request finalization |
| client -> server | ping   | Health check; server replies `pong` |
| server -> client | hello  | Connection metadata (stream id, sample rate) |
| server -> client | result | Partial or final transcript |
| server -> client | error  | Errors when the server cannot proceed |
| server -> client | pong   | Response to `ping` |
| status port -> client | status | One-shot connection count |

## Audio messages

An `audio` message can carry either base64 bytes or a list of float samples. Base64 is
recommended for efficiency.

Fields for base64 audio:

| Field | Type | Notes |
| --- | --- | --- |
| type | string | Must be `"audio"` |
| encoding | string | `"pcm16"` or `"f32"` |
| data | string | Base64-encoded bytes |
| final | boolean | End-of-stream indicator |

Encodings:

`pcm16` is little-endian signed int16 audio. `f32` is raw float32 bytes.

Example:
```json
{"type":"audio","data":"...","encoding":"pcm16","final":false}
```

Alternative, for quick debug use a float list (no base64):
```json
{"type":"audio","samples":[0.0,0.01,-0.02],"final":false}
```

## Status port (optional)

If the server is started with `--status-port`, you can open a plain TCP connection to
that port and read a single JSON line that reports the current number of connected
streams. The server closes the connection immediately after sending the response.

## Finalization behavior (current vs. upcoming)

Today, the client is responsible for ending the stream by sending either `final: true`
on the last audio message or a separate `close` message. The server will then return a
`result` with `is_final: true`.

In a future release, the server will assert and emit `is_final` automatically based on
silence detection instead of relying on the client to decide when to end the stream.

## Debugging and common errors

If the server cannot accept a stream (for example, all stream slots are in use), it
responds with an `error` message and closes the connection. You should log the error
message and retry later.
One common message in this case is `No free streaming state available.`.

If you see `{"type":"error","message":"unknown message type"}` then the payload is
missing a valid `type` or the `type` is unsupported.

If the connection closes immediately after sending audio, check these cases:

Unsupported encoding. Only `pcm16` and `f32` are accepted. Any other encoding causes the
server to log an error and close the connection.

Invalid JSON. The server ignores malformed JSON lines. A steady stream of invalid data
will look like a stalled connection.

Mismatch between port and protocol. WebSocket clients must connect to `--ws-port` and
TCP clients must connect to `--port`.

The server logs to stdout at INFO level by default. If you need more detail, adjust
the logging level in `parakeet/cli.py` or wrap the server in your own entrypoint.
