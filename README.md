# parakeet-streaming

Low-latency, real-time streaming Automatic Speech Recognition (ASR) inference service
built around NVIDIA's streaming speech models. Status: beta. This project is actively
worked on, so bugs and breaking changes are possible.

## Models

| Model name | Official model name | Parameter count | Hugging Face |
| --- | --- | --- | --- |
| small | parakeet_realtime_eou | 120M | https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 |
| large | nemotron-speech streaming | 600M | https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b |

Both models are based on the cache-aware FastConformer RNNT Architecture developed by Nvidia and designed for low latency
streaming.
The small model emits an end-of-utterance (EOU) token  and is designed for low-latency English transcription
without punctuation or
capitalization. The large model is a cache-aware FastConformer RNNT supports punctuation and capitalization
with configurable
chunk sizes for streaming workloads.

Model weights are governed by NVIDIA's Open Model License (NeMo). The model implementation in this repo
rewrites the q, k, v layers  in FastConformer to be a single linear layer ```qkv``` and
chunks this to get the query, key and values.
Because of this, weights of both models are saved in the ```safetensors``` format in a separate huggingface repo.
This allows for faster weight loading but the official
license terms still apply.

Expected latency is about 160 ms. On an NVIDIA A100, the small model supports around
80 concurrent streams and the large model supports around 60 concurrent streams.
Official benchmark scores are coming soon.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

The first run will download model weights from Hugging Face.

## Run the server

```bash
parakeet-server serve --host 0.0.0.0 --port 8765 --ws-port 8766 --device cuda
```

Common flags include `--model-size`, `--sample-rate`, `--max-num-streams`,
`--max-stream-seconds`, `--max-seq-len`, `--ws-port`, and `--status-port`.

## Client protocol

Clients can connect over WebSockets on `--ws-port` or use raw TCP on `--port`. Both
transports exchange the same JSON message shapes. WebSockets send one JSON object per
message. TCP uses newline-delimited JSON.

Audio messages look like this:
```json
{"type":"audio","data":"...","encoding":"pcm16","final":false}
```

Only `pcm16` and `f32` encodings are supported. The server responds with `result`
messages that include `text`, `token_ids`, and `is_final`.

There is no VAD integration yet, so silence is not handled automatically. Today, the
client ends a stream by sending `final: true` or `close`. In a future release, the
server will assert `final` based on silence detection instead of requiring the
client to decide when a stream is finished.
Please refer to the ```docs``` folder for more information on how to get started.
