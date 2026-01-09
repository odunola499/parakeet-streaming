import argparse
import time

import librosa
import numpy as np
import torch
import threading

from parakeet.config import Config
from parakeet.engine.asr_engine import ASREngine


def feed_engine(stream_id, audio, chunk_samples, engine, sleep_seconds: float):
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        engine.push_samples(stream_id, chunk, final=False)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    engine.push_samples(stream_id, np.empty((0,), dtype=np.float32), final=True)


def pull_from_engine(stream_id, engine, show_text: bool = True):
    while True:
        results = engine.collect_stream_results(stream_id)
        if results:
            for result in results:
                if show_text:
                    text = result.text
                    print(f"[stream {result.stream_id}] {text}", flush=True)
                if result.is_final:
                    return
        else:
            time.sleep(0.01)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-streams", type=int, default=1)
    parser.add_argument(
        "--audio",
        nargs="*",
        default=[
            "/Users/odunolajenrola/Documents/GitHub/parakeet-streaming/test.mp3",
            "/Users/odunolajenrola/Documents/GitHub/parakeet-streaming/test.mp3",
        ],
    )
    parser.add_argument("--duration", type=float, default=2)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=0.25)
    parser.add_argument("--model-size", choices=("small", "large"), default="small")
    parser.add_argument(
        "--stream-speed",
        type=float,
        default=2.0,
        help="1.0 = real-time, >1.0 = faster than real-time, <=0 = no sleep",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config(size=args.model_size)
    print("Loading model...")
    engine = ASREngine(config, device=device)

    stream_ids = [engine.create_stream() for _ in range(args.num_streams)]
    audio_paths = args.audio
    audios = []
    for idx in range(args.num_streams):
        path = audio_paths[idx % len(audio_paths)]
        audio, _ = librosa.load(path, sr=args.sample_rate, duration=args.duration)
        audios.append(audio)

    chunk_samples = int(args.chunk_seconds * args.sample_rate)
    if args.stream_speed > 0:
        sleep_seconds = args.chunk_seconds / args.stream_speed
    else:
        sleep_seconds = 0.0
    producers = [
        threading.Thread(
            target=feed_engine,
            args=(stream_id, audio, chunk_samples, engine, sleep_seconds),
        )
        for stream_id, audio in zip(stream_ids, audios)
    ]
    consumers = [
        threading.Thread(
            target=pull_from_engine, args=(stream_id, engine, not args.quiet)
        )
        for stream_id in stream_ids
    ]

    for prod, cons in zip(producers, consumers):
        prod.start()
        time.sleep(0.5)
        cons.start()

    for prod, cons in zip(producers, consumers):
        prod.join()
        cons.join()

    engine.close()


if __name__ == "__main__":
    main()
