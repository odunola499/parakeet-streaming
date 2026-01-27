# Python microphone example

This is an end-to-end microphone demo example for the streaming server. The
server should already be running with WebSockets enabled, for example:

```bash
parakeet-server serve --host 0.0.0.0 --port 8765 --ws-port 8000 --device cuda
```

Install dependencies:
```bash
pip install websockets numpy sounddevice
```

Run the script, speak into the mic, then press Ctrl+C to end the stream. The client is
responsible for sending the final marker, so the example sends `final: true` when you
stop it.

```python
import asyncio
import base64
import json

import numpy as np
import sounddevice as sd
import websockets

HOST, PORT = "127.0.0.1", 8000
SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.25

async def main() -> None:
    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            return
        samples = indata[:, 0].astype(np.float32, copy=True)
        loop.call_soon_threadsafe(audio_queue.put_nowait, samples)

    async with websockets.connect(f"ws://{HOST}:{PORT}") as ws:
        hello = json.loads(await ws.recv())
        print("hello:", hello)

        stop_event = asyncio.Event()

        async def sender():
            blocksize = max(1, int(SAMPLE_RATE * CHUNK_SECONDS))
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            ):
                while not stop_event.is_set():
                    chunk = await audio_queue.get()
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

            await ws.send(
                json.dumps({"type": "audio", "data": "", "encoding": "pcm16", "final": True})
            )

        async def receiver():
            while not stop_event.is_set():
                msg = json.loads(await ws.recv())
                if msg.get("type") == "result":
                    print(msg.get("text", ""))
                    if msg.get("is_final"):
                        stop_event.set()

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        try:
            await asyncio.gather(send_task, recv_task)
        except KeyboardInterrupt:
            stop_event.set()
            await send_task

asyncio.run(main())
```
You should see what you say get printed on your terminal in realtime.
