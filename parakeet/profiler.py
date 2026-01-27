from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import threading

import librosa
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile as torch_profile, record_function

from parakeet.config import Config
from parakeet.engine.asr_engine import ASREngine

DEFAULT_AUDIO = Path(__file__).resolve().parents[1] / "audio" / "test.mp3"


def _resolve_audio_paths(paths: list[str] | None) -> list[Path]:
    if paths:
        return [Path(p) for p in paths]
    if DEFAULT_AUDIO.exists():
        return [DEFAULT_AUDIO]
    return []


def _load_audio(path: Path | None, sample_rate: int, duration: float) -> np.ndarray:
    if path and path.exists():
        audio, _ = librosa.load(
            path.as_posix(), sr=sample_rate, duration=duration, mono=True
        )
        return np.asarray(audio, dtype=np.float32)
    length = int(duration * sample_rate)
    if length <= 0:
        return np.empty((0,), dtype=np.float32)
    t = np.linspace(0.0, duration, num=length, endpoint=False, dtype=np.float32)
    tone = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    return tone.astype(np.float32, copy=False)


def _load_audios(
    audio_paths: list[str] | None,
    sample_rate: int,
    duration: float,
    num_streams: int,
) -> list[np.ndarray]:
    resolved = _resolve_audio_paths(audio_paths)
    audios: list[np.ndarray] = []
    if resolved:
        for idx in range(num_streams):
            path = resolved[idx % len(resolved)]
            audios.append(_load_audio(path, sample_rate, duration))
        return audios
    for _ in range(num_streams):
        audios.append(_load_audio(None, sample_rate, duration))
    return audios


def _feed_engine(
    stream_id: int, audio: np.ndarray, chunk_samples: int, engine: ASREngine
) -> None:
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples]
        engine.push_samples(stream_id, chunk, final=False)
    engine.push_samples(stream_id, np.empty((0,), dtype=np.float32), final=True)


def _consume_engine(
    stream_id: int,
    engine: ASREngine,
    show_text: bool,
    poll_interval: float,
    deadline: float,
    finished: threading.Event,
) -> None:
    while True:
        results = engine.collect_stream_results(stream_id)
        if results:
            for result in results:
                if show_text:
                    print(f"[stream {result.stream_id}] {result.text}", flush=True)
                if result.is_final:
                    finished.set()
                    return
        else:
            if time.monotonic() >= deadline:
                finished.set()
                return
            time.sleep(poll_interval)


def _run_streams(
    engine: ASREngine,
    audios: list[np.ndarray],
    chunk_samples: int,
    show_text: bool,
    poll_interval: float,
    timeout_seconds: float,
) -> None:
    stream_ids = [engine.create_stream() for _ in audios]
    done_events = [threading.Event() for _ in stream_ids]
    deadline = time.monotonic() + timeout_seconds

    producers = [
        threading.Thread(
            target=_feed_engine,
            args=(stream_id, audio, chunk_samples, engine),
        )
        for stream_id, audio in zip(stream_ids, audios)
    ]
    consumers = [
        threading.Thread(
            target=_consume_engine,
            args=(
                stream_id,
                engine,
                show_text,
                poll_interval,
                deadline,
                done_event,
            ),
        )
        for stream_id, done_event in zip(stream_ids, done_events)
    ]

    for thread in producers + consumers:
        thread.start()
    for thread in producers + consumers:
        thread.join()


def _driver_step(engine: ASREngine) -> bool:
    runner = engine.runner
    with record_function("pipeline_step"):
        runner._turn_detection_step()
        results = runner.step()

    result_ids = set()
    for result in results:
        runner._results_queue.put(result)
        result_ids.add(result.seq_id)

    finalized = runner._finalize_sequences(result_ids)
    if results or finalized:
        runner._notify_update()
    return bool(results or finalized)


def _run_streams_single_thread(
    engine: ASREngine,
    audios: list[np.ndarray],
    chunk_samples: int,
    show_text: bool,
    poll_interval: float,
    timeout_seconds: float,
) -> None:
    stream_ids = [engine.create_stream() for _ in audios]
    offsets = [0 for _ in stream_ids]
    finals_sent = [False for _ in stream_ids]
    done = [False for _ in stream_ids]
    deadline = time.monotonic() + timeout_seconds

    while not all(done):
        if time.monotonic() >= deadline:
            break

        any_pushed = False
        for idx, (stream_id, audio) in enumerate(zip(stream_ids, audios)):
            if done[idx]:
                continue
            if offsets[idx] < len(audio):
                end = min(offsets[idx] + chunk_samples, len(audio))
                chunk = audio[offsets[idx] : end]
                engine.push_samples(stream_id, chunk, final=False)
                offsets[idx] = end
                any_pushed = True
            elif not finals_sent[idx]:
                engine.push_samples(
                    stream_id, np.empty((0,), dtype=np.float32), final=True
                )
                finals_sent[idx] = True
                any_pushed = True

        progressed = _driver_step(engine)
        any_results = False
        for idx, stream_id in enumerate(stream_ids):
            if done[idx]:
                continue
            results = engine.collect_stream_results(stream_id)
            if results:
                any_results = True
                for result in results:
                    if show_text:
                        print(f"[stream {result.stream_id}] {result.text}", flush=True)
                    if result.is_final:
                        done[idx] = True

        if not (any_pushed or progressed or any_results):
            time.sleep(poll_interval)


def _format_metrics(metrics: dict[str, object]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)


def _print_bottleneck_summary(metrics: dict[str, object], run_time: float) -> None:
    timings = metrics.get("timings_ms", {})
    counts = metrics.get("counts", {})
    run_ms = max(run_time * 1000.0, 0.0)

    stages = [
        ("encode", "encode_avg", "encode_calls"),
        ("pre_encode", "pre_encode_avg", "pre_encode_calls"),
        ("decode", "decode_avg", "decode_calls"),
        ("turn_detection", "turn_detection_avg", "turn_detection_calls"),
    ]

    rows = []
    accounted_ms = 0.0
    for name, avg_key, count_key in stages:
        avg_ms = float(timings.get(avg_key, 0.0))
        calls = int(counts.get(count_key, 0))
        total_ms = avg_ms * calls
        accounted_ms += total_ms
        pct = (total_ms / run_ms * 100.0) if run_ms > 0 else 0.0
        rows.append((name, total_ms, avg_ms, calls, pct))

    overhead_ms = max(run_ms - accounted_ms, 0.0)
    if overhead_ms > 0.0:
        pct = (overhead_ms / run_ms * 100.0) if run_ms > 0 else 0.0
        rows.append(("other/overhead", overhead_ms, 0.0, 0, pct))

    rows.sort(key=lambda row: row[1], reverse=True)

    print("\nBottleneck summary (engine timings)")
    header = (
        f"{'stage':<16} {'total_ms':>12} {'avg_ms':>10} {'calls':>8} {'pct_run':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, total_ms, avg_ms, calls, pct in rows:
        print(f"{name:<16} {total_ms:>12.2f} {avg_ms:>10.2f} {calls:>8d} {pct:>7.2f}%")


def _configure_single_thread() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def run_profile(args: argparse.Namespace) -> None:
    if args.single_thread:
        _configure_single_thread()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    max_stream_seconds = args.max_stream_seconds
    if max_stream_seconds is None:
        max_stream_seconds = max(5, int(args.duration) + 2)

    config = Config(
        size=args.model_size,
        max_num_streams=max(args.max_num_streams, args.num_streams),
        sample_rate=args.sample_rate,
        max_stream_seconds=max_stream_seconds,
    )

    print("Loading model...")
    load_start = time.monotonic()
    engine = ASREngine(config, device=device)
    load_time = time.monotonic() - load_start
    if args.single_thread:
        engine.runner.stop_workers()

    audios = _load_audios(args.audio, args.sample_rate, args.duration, args.num_streams)
    chunk_samples = max(1, int(args.chunk_seconds * args.sample_rate))
    total_audio_sec = sum(len(a) / args.sample_rate for a in audios)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("Running profiling pass...")
    run_start = time.monotonic()
    with torch_profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        if args.single_thread:
            _run_streams_single_thread(
                engine,
                audios,
                chunk_samples,
                args.show_text,
                args.poll_interval,
                args.timeout_seconds,
            )
        else:
            _run_streams(
                engine,
                audios,
                chunk_samples,
                args.show_text,
                args.poll_interval,
                args.timeout_seconds,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    run_time = time.monotonic() - run_start

    metrics = engine.get_metrics()
    engine.close()

    trace_path = Path(args.trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(trace_path.as_posix())

    print("\nProfile summary")
    print(f"  device: {device}")
    print(f"  model_size: {args.model_size}")
    print(f"  num_streams: {args.num_streams}")
    print(f"  sample_rate: {args.sample_rate}")
    print(f"  chunk_seconds: {args.chunk_seconds}")
    print(f"  audio_seconds_total: {total_audio_sec:.2f}")
    print(f"  load_time_sec: {load_time:.2f}")
    print(f"  run_time_sec: {run_time:.2f}")
    if total_audio_sec > 0:
        print(f"  realtime_factor: {run_time / total_audio_sec:.2f}x")

    if device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated()
        print(f"  max_cuda_mem_bytes: {max_mem}")

    print("\nEngine metrics")
    print(_format_metrics(metrics))

    _print_bottleneck_summary(metrics, run_time)

    print("\nTop operators by CPU time")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=args.top))

    if device.type == "cuda":
        print("\nTop operators by CUDA time")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.top))

    print(f"\nChrome trace saved to {trace_path.as_posix()}")


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-size", choices=("small", "large"), default="small")
    parser.add_argument("--num-streams", type=int, default=1)
    parser.add_argument("--max-num-streams", type=int, default=1)
    parser.add_argument("--audio", nargs="*")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=0.25)
    parser.add_argument("--poll-interval", type=float, default=0.01)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--trace-path", default="traces/parakeet_profile.json")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--max-stream-seconds", type=int)
    parser.add_argument(
        "--single-thread",
        action="store_true",
        help="Run the pipeline driver in a single thread for cleaner traces.",
    )
