# TypeScript client quickstart

This quickstart shows common client scenarios for both WebSockets and raw TCP. The
server must already be running.

Example server command:
```bash
parakeet-server serve --host 0.0.0.0 --port 8765 --ws-port 8766 --device cuda
```

To verify the server is reachable, connect and read the `hello` message, then send a
`ping` and confirm you get a `pong`. If you see the `hello` printed in the examples
below, your connection is working.

## Scenario 1: Browser WebSocket microphone streaming (TypeScript)

This is the simplest browser example. It uses the Web Audio API to capture audio and
sends PCM16 chunks over WebSockets. This uses `ScriptProcessorNode` for simplicity. For
production, use `AudioWorklet` instead.

```ts
const ws = new WebSocket("ws://127.0.0.1:8766");

ws.addEventListener("message", (event) => {
  const msg = JSON.parse(event.data as string);
  if (msg.type === "hello") {
    console.log("hello:", msg);
  }
  if (msg.type === "result") {
    console.log(msg.text);
  }
});

ws.addEventListener("open", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const pcm16 = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      pcm16[i] = Math.round(s * 32767);
    }
    const base64 = btoa(String.fromCharCode(...new Uint8Array(pcm16.buffer)));
    ws.send(
      JSON.stringify({ type: "audio", data: base64, encoding: "pcm16", final: false })
    );
  };

  source.connect(processor);
  processor.connect(audioCtx.destination);
});
```

When you stop recording, send a final marker so the server can finalize the stream:
```ts
ws.send(JSON.stringify({ type: "audio", data: "", encoding: "pcm16", final: true }));
```

## Scenario 2: Node raw TCP smoke test

This is the smallest useful TCP example. It sends 250 ms of silence and waits for a
final result. Use this to verify connectivity without worrying about audio capture.

```ts
import net from "node:net";

const sock = net.createConnection(8765, "127.0.0.1");
let buffer = "";

function sendJson(payload: unknown) {
  sock.write(JSON.stringify(payload) + "\n");
}

sock.on("data", (data) => {
  buffer += data.toString("utf8");
  let idx;
  while ((idx = buffer.indexOf("\n")) !== -1) {
    const line = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 1);
    if (!line.trim()) continue;
    const msg = JSON.parse(line);
    if (msg.type === "hello") {
      console.log("hello:", msg);
    }
    if (msg.type === "result") {
      console.log(msg.text);
      if (msg.is_final) {
        sock.end();
      }
    }
  }
});

sock.on("connect", () => {
  const pcm16 = Buffer.alloc(16000 * 2 / 4); // 250ms of silence at 16kHz
  sendJson({
    type: "audio",
    data: pcm16.toString("base64"),
    encoding: "pcm16",
    final: true,
  });
});
```
