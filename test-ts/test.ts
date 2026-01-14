import net from "node:net";

const sock = net.createConnection(8765, "149.36.1.129");
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
    if (msg.type === "pong") {
      console.log("pong received");
    }
  }
});

sock.on("connect", () => {
  console.log("Connected to server");
  // Send ping first
  sendJson({ type: "ping" });
  
  // Wait a bit then send audio
  setTimeout(() => {
    const pcm16 = Buffer.alloc((16000 * 2) / 4); // 250ms of silence at 16kHz
    sendJson({
      type: "audio",
      data: pcm16.toString("base64"),
      encoding: "pcm16",
      final: true,
    });
  }, 100);
});

sock.on("error", (err) => {
  console.error("Socket error:", err);
});

sock.on("close", () => {
  console.log("Connection closed");
  process.exit(0);
});
