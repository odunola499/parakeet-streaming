// WebSocket test for Node
import WebSocket from "ws";

const ws = new WebSocket("ws://149.36.1.129:8766");

ws.on("message", (data) => {
  const msg = JSON.parse(data.toString());
  if (msg.type === "hello") {
    console.log("hello:", msg);
  }
  if (msg.type === "result") {
    console.log("result:", msg.text);
    if (msg.is_final) {
      ws.close();
    }
  }
  if (msg.type === "pong") {
    console.log("pong received");
  }
});

ws.on("open", () => {
  console.log("WebSocket connected");
  ws.send(JSON.stringify({ type: "ping" }));
  
  setTimeout(() => {
    const pcm16 = Buffer.alloc((16000 * 2) / 4); // 250ms of silence at 16kHz
    ws.send(
      JSON.stringify({
        type: "audio",
        data: pcm16.toString("base64"),
        encoding: "pcm16",
        final: true,
      })
    );
  }, 100);
});

ws.on("error", (err) => {
  console.error("WebSocket error:", err);
});

ws.on("close", () => {
  console.log("WebSocket closed");
  process.exit(0);
});
