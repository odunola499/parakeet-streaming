import time

import librosa
import numpy as np
import torch
from tqdm.auto import tqdm
import threading


from parakeet.config import Config
from parakeet.engine.asr_engine import ASREngine


def to_pcm_bytes(samples: np.ndarray) -> bytes:
    scaled = np.clip(samples, -1.0, 1.0)
    pcm = (scaled * 32767.0).astype(np.int16)
    return pcm.tobytes()


def feed_engine(stream_id, audio, chunk_samples, engine):
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        engine.push_samples(stream_id, chunk, final=False)
        time.sleep(0.01)
    engine.push_samples(stream_id, np.empty((0,), dtype=np.float32), final=True)


def pull_from_engine(stream_id, engine, pbar):
    while True:
        results = engine.collect_results()
        if results:
            result = [i for i in results if i.stream_id == stream_id][0]
            text = result.text
            print(text)
            print("")
            pbar.update(1)
        else:
            time.sleep(0.1)
        if engine.streams[stream_id].is_finished:
            pbar.close()
            break


def main():
    first_path = "/Users/odunolajenrola/Documents/GitHub/parakeet-streaming/test.mp3"
    second_path = "/Users/odunolajenrola/Documents/GitHub/parakeet-streaming/test_2.mp3"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = Config(size="small")
    print("Loading model...")
    engine = ASREngine(config, device=device)
    first_stream_id = engine.create_stream()
    second_stream_id = engine.create_stream()

    first_audio, _ = librosa.load(first_path, sr=16000)
    second_audio, _ = librosa.load(second_path, sr=16000)

    chunk_samples = int(0.25 * 16000)
    first_pbar = tqdm(desc=f"Running stream{first_stream_id}...")
    second_pbar = tqdm(desc=f"Running stream{second_stream_id}...")

    producer_one = threading.Thread(
        target=feed_engine,
        args=(first_stream_id, first_audio, chunk_samples, engine),
    )
    producer_two = threading.Thread(
        target=feed_engine,
        args=(second_stream_id, second_audio, chunk_samples, engine),
    )
    consumer_one = threading.Thread(
        target=pull_from_engine, args=(first_stream_id, engine, first_pbar)
    )
    consumer_two = threading.Thread(
        target=pull_from_engine, args=(second_stream_id, engine, second_pbar)
    )

    producer_one.start()
    consumer_one.start()
    producer_two.start()
    consumer_two.start()

    producer_two.join()
    consumer_two.join()
    consumer_one.join()
    producer_one.join()

    engine.close()


if __name__ == "__main__":
    main()
