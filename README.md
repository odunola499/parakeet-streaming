# parakeet-streaming

Low-latency, real-time streaming Automatic Speech Recognition (ASR) inference service
built around NVIDIA's streaming speech models. This project is still actively
worked on, so bugs and breaking changes are possible.
Read more about the project [here!!](https://odunola.bearblog.dev/parakeet-streaming/)

## Models

| Model name | Official model name | Parameter count | Hugging Face |
| --- | --- | --- | --- |
| small | parakeet_realtime_eou | 120M | https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1 |
| large | nemotron-speech streaming | 600M | https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b |

Both models are based on the cache-aware FastConformer RNNT Architecture developed by Nvidia and designed for low latency
streaming.
The small model emits an end-of-utterance (EOU) token  and is designed for low-latency English transcription
without punctuation or
capitalization. The large model supports punctuation and capitalization.

Model weights are governed by NVIDIA's Open Model License (NeMo).
The model weights of both models are also saved in the ```safetensors``` format in a separate huggingface repo for
easier loading.
This also allows for faster weight loading but the official NeMO
license terms still apply for the model checkpoints.

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
parakeet-server serve --host 0.0.0.0 --port 8765 --ws-port 8000 --device cuda
```

Common flags include `--model-size`, `--sample-rate`, `--max-num-streams`,
`--max-stream-seconds`, `--ws-port`, and `--status-port`.

## Client protocol

Clients can connect over WebSockets on `--ws-port` or use raw TCP on `--port`. Both
transports exchange the same JSON message shapes. WebSockets send one JSON object per
message. TCP uses newline-delimited JSON.

Audio messages look like this:
```json
{"type":"audio","data":"...","encoding":"pcm16","final":false}
```

Only `pcm16` and `f32` encodings are supported. The server responds with `result`
messages that include `text`, `token_ids`, `confidence_scores`, and `is_final`.
`confidence_scores` are aligned with `token_ids` for the newly emitted tokens.

There is no VAD integration yet, so silence is not handled automatically. Today, the
client ends a stream by sending `final: true` or `close`, or closing the connection.
Please refer to the ```docs``` folder for more information on how to get started,
including Python and TS client quickstarts)
