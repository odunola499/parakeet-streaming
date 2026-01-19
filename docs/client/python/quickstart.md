# Python client quickstart

This quickstart shows a few common client scenarios. The server must already be
running. If you want WebSockets, start the server with `--ws-port` and connect to that
port.

Example server command:
```bash
parakeet-server serve --host 0.0.0.0 --port 8765 --ws-port 8000 --device cuda
```

To verify the server is reachable, connect and read the `hello` message, then send a
`ping` and confirm you get a `pong`. The examples below already do the `hello` read,
so if you see that printed you are connected correctly.

The snippets below use `numpy` and `soundfile` for convenience. If you only need raw
TCP and already have PCM bytes, you can skip those dependencies.

For WebSocket examples, you also need the `websockets` package:
```bash
pip install websockets numpy soundfile
```

## Scenario 1: WebSocket streaming from a WAV file (PCM16)

This example sends 250 ms chunks over WebSockets and prints partial/final transcripts.

```python
import asyncio
import base64
import json

import numpy as np
import soundfile as sf
import websockets

HOST, PORT = "127.0.0.1", 8000
AUDIO_PATH = "./test.wav"
CHUNK_SECONDS = 0.25

async def main():
    uri = f"ws://{HOST}:{PORT}"
    async with websockets.connect(uri) as ws:
        hello = json.loads(await ws.recv())
        sample_rate = int(hello["sample_rate"])
        print("hello:", hello)

        audio, sr = sf.read(AUDIO_PATH, dtype="float32")
        if sr != sample_rate:
            raise ValueError(f"sample rate mismatch: got {sr}, expected {sample_rate}")
        if audio.ndim > 1:
            audio = audio[:, 0]

        chunk_samples = int(CHUNK_SECONDS * sample_rate)
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            pcm16 = np.clip(chunk, -1.0, 1.0)
            pcm16 = (pcm16 * 32767.0).astype(np.int16)
            await ws.send(
                json.dumps(
                    {
                        "type": "audio",
                        "data": base64.b64encode(pcm16.tobytes()).decode("ascii"),
                        "encoding": "pcm16",
                        "final": False,
                    }
                )
            )

        # Finalize the stream.
        await ws.send(
            json.dumps({"type": "audio", "data": "", "encoding": "pcm16", "final": True})
        )

        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "result":
                print(msg.get("text", ""))
                if msg.get("is_final"):
                    break

asyncio.run(main())
```

Result messages also include `token_ids` and `confidence_scores`, aligned to the
newly emitted tokens.

## Scenario 2: Raw TCP smoke test (send silence, then finalize)

This is the smallest useful TCP example. It sends 250 ms of silence and waits for a
final result. Use this to verify connectivity without worrying about audio capture.

```python
import base64
import json
import socket

import numpy as np

HOST, PORT = "127.0.0.1", 8765
sock = socket.create_connection((HOST, PORT))
buffer = b""

def read_message():
    global buffer
    while b"\n" not in buffer:
        data = sock.recv(4096)
        if not data:
            return None
        buffer += data
    line, buffer = buffer.split(b"\n", 1)
    return json.loads(line.decode("utf-8"))

def send_json(payload):
    message = json.dumps(payload, separators=(",", ":")) + "\n"
    sock.sendall(message.encode("utf-8"))

hello = read_message()
print("hello:", hello)

samples = np.zeros(int(0.25 * 16000), dtype=np.float32)
pcm16 = (samples * 32767.0).astype(np.int16)
send_json(
    {
        "type": "audio",
        "data": base64.b64encode(pcm16.tobytes()).decode("ascii"),
        "encoding": "pcm16",
        "final": True,
    }
)

while True:
    msg = read_message()
    if msg is None:
        break
    if msg.get("type") == "result":
        print(msg.get("text", ""))
        if msg.get("is_final"):
            break

sock.close()
```

## Scenario 3: Explicit close message

If you need to end a stream without sending more audio, send `close`. The server will
finalize the stream and return a final result as soon as decoding completes.

```python
send_json({"type": "close"})
```

## Scenario 4: Send float32 samples directly (debug only)

For quick testing you can send a float list. This is convenient but inefficient for
real-time workloads.

```python
send_json({"type": "audio", "samples": [0.0, 0.01, -0.02], "final": True})
```
