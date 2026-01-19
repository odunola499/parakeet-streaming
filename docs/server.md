# Server guide

This guide covers getting started with the service including starting it up, understanding the ports and
handling incoming streams.


## Quick start

Create a virtualenv, install dependencies, and run the server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

parakeet-server serve --host 0.0.0.0 --port 8765 --ws-port 8000 --device cuda
```

The first run downloads model weights from Hugging Face. Subsequent runs reuse the
cached weights.

## Ports and protocols

The server can listen on multiple ports at once. Each port has a specific purpose:

`--port` is the raw TCP port. Clients send newline-delimited JSON.

`--ws-port` is the WebSocket port. Clients send one JSON object per WebSocket message.

`--status-port` is a one-shot status endpoint that reports the number of connected
streams, then closes the connection.

If you are building a browser client, you must use `--ws-port`. If you are building a
backend or service client, both TCP and WebSockets are valid.

## Stream lifecycle

Each connection creates a stream in the engine. The server sends a `hello` message that
includes a `stream_id` and `sample_rate`. Clients then send `audio` messages until they
choose to finalize the stream. Finalization happens when the client sends a message with
`final: true` or sends a separate `close` message. The server emits `result` messages as
it decodes, and the final transcript is marked with `is_final: true`. Each `result`
includes `text`, `token_ids`, and `confidence_scores` (aligned to the newly emitted
tokens).

A client can send `ping` to verify liveness. The server replies with `pong`.

## Configuration flags

These are the most commonly used flags when starting the server:

`--model-size` selects the model variant, `small` or `large`.

`--device` selects the PyTorch device, typically `cpu` or `cuda`.

`--sample-rate` sets the expected sample rate for audio input. Clients should match it. 16000 by default.

`--max-num-streams` sets the maximum number of concurrent streams.

`--max-stream-seconds` caps how much audio a stream can buffer.

You can view the full list of flags with:

```bash
parakeet-server --help
```

## Handling incoming streams

Incoming audio is buffered per stream. The engine then processes audio in small chunks
and emits partial results as decoding progresses. Streams are processed concurrently
until `max_num_streams` is reached, at which point new connections will receive an error
message indicating there is no free streaming state.

If a client disconnects abruptly, the server will try to finalize and clean up the
stream, but partial results may be lost. For best results, always send `final: true` or
`close` when you are done.

## Status checks

If you enable `--status-port`, you can connect to that port and read a single JSON line
like this:

```json
{"type":"status","connected_streams":3}
```

The connection closes immediately after that response. This endpoint is intended for
simple health checks and dashboards.

## Operational notes

The server logs to stdout at INFO level by default. If you want more detail, adjust the
logging configuration in `parakeet/cli.py` or wrap the server in your own entrypoint.

There is currently no VAD integration. Silence is not handled automatically, so clients
must choose when to finalize a stream.
