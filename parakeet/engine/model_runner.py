from dataclasses import dataclass
from pathlib import Path
import queue
import threading

import numpy as np
import torch
import time
from torch.profiler import record_function

from parakeet.config import Config
from parakeet.engine.scheduler import Scheduler
from parakeet.engine.sequence import Sequence, SequenceStatus
from parakeet.engine.turn_detection import TurnDetection
from parakeet.model import Parakeet
from parakeet.kernels.paged_attention import paged_attention_status


@dataclass
class DecodeResult:
    seq_id: int
    token_ids: list[int]
    confidence_scores: list[float]


@dataclass
class RunnerMetrics:
    start_time: float
    td_time: float = 0.0
    td_calls: int = 0
    td_sequences: int = 0
    pre_encode_time: float = 0.0
    pre_encode_calls: int = 0
    pre_encode_chunks: int = 0
    encode_time: float = 0.0
    encode_calls: int = 0
    encode_frames: int = 0
    decode_time: float = 0.0
    decode_calls: int = 0
    decode_frames: int = 0
    decode_tokens: int = 0


class ShapeLogger:
    def __init__(self, path: str | None, max_entries: int = 200):
        self._path = Path(path) if path else None
        self._max_entries = max_entries
        self._seen: set[tuple] = set()
        self._lock = threading.Lock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, kind: str, **items: object) -> None:
        if self._path is None:
            return
        key = (kind, tuple(items.items()))
        with self._lock:
            if key in self._seen:
                return
            if len(self._seen) >= self._max_entries:
                return
            self._seen.add(key)
            line = kind + " " + " ".join(f"{k}={v}" for k, v in items.items()) + "\n"
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)


@dataclass
class EncodeGraph:
    batch_size: int
    batch: torch.Tensor
    lengths: torch.Tensor
    slot_index: torch.Tensor
    pad_mask: torch.Tensor
    attn_mask: torch.Tensor
    batch_state: object
    graph: torch.cuda.CUDAGraph
    out: torch.Tensor


class ModelRunner:
    def __init__(self, config: Config, device: torch.device, scheduler: Scheduler):
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self._amp_enabled = False
        self._amp_dtype: torch.dtype | None = None
        self._shape_logger = ShapeLogger(self.config.shape_log_path)
        self._encode_graphs: dict[int, EncodeGraph] = {}

        self._init_model()
        self._paged_attention = False
        if self.config.paged_kv_cache:
            enabled, reason = paged_attention_status(self.device)
            if enabled:
                self._paged_attention = True
            else:
                print(f"Paged attention disabled ({reason}); using contiguous KV cache.")
                self.config.paged_kv_cache = False
        self.scheduler.init_state_pool(
            self.encoder,
            self.device,
            self._dtype,
            pool_size=config.max_num_streams,
            paged_kv_cache=config.paged_kv_cache,
            paged_kv_page_size=config.paged_kv_page_size,
            paged_kv_max_pages=config.paged_kv_max_pages,
            paged_attention=self._paged_attention,
        )
        self._init_encode_graphs()

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._results_queue: queue.Queue[DecodeResult] = queue.Queue()
        self._td_ready_queue: queue.SimpleQueue[Sequence] = queue.SimpleQueue()
        self._pre_encode_ready_queue: queue.SimpleQueue[Sequence] = queue.SimpleQueue()
        self._encode_ready_queue: queue.SimpleQueue[Sequence] = queue.SimpleQueue()
        self._decode_ready_queue: queue.SimpleQueue[Sequence] = queue.SimpleQueue()
        self._queue_depths = {"pre_encode": 0, "encode": 0, "decode": 0}
        self._queue_depth_lock = threading.Lock()
        self._update_cond = threading.Condition()
        self._update_seq = 0
        self._metrics = RunnerMetrics(start_time=time.monotonic())
        self._metrics_lock = threading.Lock()

    def _init_model(self) -> None:
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except AttributeError:
                pass
        dtype = None
        if self.config.dtype == "fp16":
            dtype = torch.float16
        elif self.config.dtype == "bf16":
            dtype = torch.bfloat16
        if dtype is not None and self.device.type != "cuda":
            print(f"Requested dtype {self.config.dtype} on CPU; using fp32.")
            dtype = None

        self.model = Parakeet.from_pretrained(self.config.size, dtype=dtype).to(
            self.device
        )
        self.model.eval()
        self.turn_detection = TurnDetection()

        self.encoder = self.model.encoder
        self.pre_encoder = self.encoder.pre_encode
        self.decoder = self.model.predictor
        self.joiner = self.model.joiner

        param = next(self.model.parameters())
        self._dtype = param.dtype
        if dtype is not None and self.device.type == "cuda":
            self._amp_enabled = True
            self._amp_dtype = dtype

        self._subsampling_factor = self.model.config.subsampling_factor
        self._lookahead = self.encoder.att_context_size[1]
        self._chunk_frames_first = 1 + self._subsampling_factor * self._lookahead
        self._chunk_frames_next = (
            self._subsampling_factor + self._subsampling_factor * self._lookahead
        )
        self._feature_extractor = self.model._feature_extractor
        self._samples_per_frame = self._feature_extractor.hop_length
        self.blank_id = self.model.blank_id
        self._max_symbols_per_frame = 5
        self.start_ids = torch.full(
            (1,), self.blank_id, dtype=torch.long, device=self.device
        )

    def _init_encode_graphs(self) -> None:
        if not self.config.cuda_graphs:
            return
        if self.config.paged_kv_cache:
            print("CUDA graphs disabled when paged KV cache is enabled.")
            return
        if self.device.type != "cuda":
            print("CUDA graphs requested but device is not CUDA; skipping.")
            return
        if self.config.dtype != "fp32":
            print("CUDA graphs only enabled for fp32 in this build; skipping.")
            return

        batch_sizes = sorted({int(bs) for bs in self.config.cuda_graph_batch_sizes if bs})
        if not batch_sizes:
            return

        max_len = self._lookahead + 1
        hidden = self.model.config.enc_hidden_dim
        key_len_max = self.encoder.att_context_size[0] + max_len

        try:
            for bs in batch_sizes:
                batch = torch.zeros(
                    (bs, max_len, hidden), device=self.device, dtype=self._dtype
                )
                lengths = torch.full(
                    (bs,), max_len, device=self.device, dtype=torch.int64
                )
                slot_index = torch.zeros((bs,), device=self.device, dtype=torch.long)
                pad_mask = torch.zeros(
                    (bs, max_len), device=self.device, dtype=torch.bool
                )
                attn_mask = torch.zeros(
                    (bs, max_len, key_len_max),
                    device=self.device,
                    dtype=torch.bool,
                )
                batch_state = self.encoder.init_streaming_state(
                    batch_size=bs, device=self.device, dtype=self._dtype
                )

                with torch.inference_mode():
                    _ = self.encoder.forward_with_masks(
                        batch, batch_state, pad_mask, attn_mask, length=lengths
                    )
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out, _ = self.encoder.forward_with_masks(
                        batch, batch_state, pad_mask, attn_mask, length=lengths
                    )

                self._encode_graphs[bs] = EncodeGraph(
                    batch_size=bs,
                    batch=batch,
                    lengths=lengths,
                    slot_index=slot_index,
                    pad_mask=pad_mask,
                    attn_mask=attn_mask,
                    batch_state=batch_state,
                    graph=graph,
                    out=out,
                )
        except Exception as exc:
            print(f"CUDA graph init failed; disabling graphs. ({exc})")
            self._encode_graphs = {}

    def create_sequence(self) -> Sequence | None:
        encoder_state_id = self.scheduler.acquire_state_slot()
        if encoder_state_id is None:
            return None

        pred_state = self.decoder.init_state(1)
        if isinstance(pred_state, tuple):
            pred_state = tuple(s.to(self.device) for s in pred_state)
        else:
            pred_state = pred_state.to(self.device)

        with torch.inference_mode():
            pred_out, pred_state = self.decoder(self.start_ids, state=pred_state)

        pre_encode_cache_size = self.encoder.pre_encode_cache_size[1]
        drop_extra_pre_encoded = self.encoder.drop_extra_pre_encoded
        enc_chunk_size = self._lookahead + 1

        max_pending_samples = int(
            self.config.max_stream_seconds * self.config.sample_rate
        )
        return Sequence(
            enc_hidden_dim=self.model.config.enc_hidden_dim,
            enc_chunk_size=enc_chunk_size,
            encoder_state=encoder_state_id,
            pred_state=pred_state,
            pred_out=pred_out,
            pre_encode_cache_size=pre_encode_cache_size,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            chunk_samples_first=self._chunk_frames_first * self._samples_per_frame,
            chunk_samples_next=self._chunk_frames_next * self._samples_per_frame,
            max_pending_samples=max_pending_samples,
            device=self.device,
        )

    def add_sequence(self, seq: Sequence) -> None:
        self.scheduler.add(seq)

    def step(self) -> list[DecodeResult]:
        self._encode_step()
        return self._decode_step()

    def start_workers(self, interval: float = 0.001) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._encode_loop, args=(interval,), daemon=True),
            threading.Thread(target=self._decode_loop, args=(interval,), daemon=True),
            threading.Thread(
                target=self._turn_detection_loop, args=(interval,), daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop_workers(self) -> None:
        if not self._threads:
            return
        self._stop_event.set()
        for thread in self._threads:
            thread.join()
        self._threads = []

    def drain_results(self) -> list[DecodeResult]:
        results: list[DecodeResult] = []
        while True:
            try:
                results.append(self._results_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def get_update_seq(self) -> int:
        with self._update_cond:
            return self._update_seq

    def wait_for_update(self, last_seq: int, timeout: float | None = None) -> int:
        with self._update_cond:
            if self._update_seq != last_seq:
                return self._update_seq
            self._update_cond.wait(timeout=timeout)
            return self._update_seq

    def get_metrics(self) -> dict[str, float | int | dict[str, float | int]]:
        with self._metrics_lock:
            metrics = RunnerMetrics(**vars(self._metrics))
        uptime = max(1e-6, time.monotonic() - metrics.start_time)
        scheduler_counts = self.scheduler.counts()
        with self._queue_depth_lock:
            queue_depths = dict(self._queue_depths)
        return {
            "uptime_sec": uptime,
            "scheduler": scheduler_counts,
            "ready_queues": {
                "pre_encode": queue_depths.get("pre_encode", 0),
                "encode": queue_depths.get("encode", 0),
                "decode": queue_depths.get("decode", 0),
            },
            "timings_ms": {
                "turn_detection_avg": (
                    (metrics.td_time / metrics.td_calls) * 1000.0
                    if metrics.td_calls
                    else 0.0
                ),
                "pre_encode_avg": (
                    (metrics.pre_encode_time / metrics.pre_encode_calls) * 1000.0
                    if metrics.pre_encode_calls
                    else 0.0
                ),
                "encode_avg": (
                    (metrics.encode_time / metrics.encode_calls) * 1000.0
                    if metrics.encode_calls
                    else 0.0
                ),
                "decode_avg": (
                    (metrics.decode_time / metrics.decode_calls) * 1000.0
                    if metrics.decode_calls
                    else 0.0
                ),
            },
            "rates_per_sec": {
                "turn_detection_sequences": metrics.td_sequences / uptime,
                "pre_encode_chunks": metrics.pre_encode_chunks / uptime,
                "encode_frames": metrics.encode_frames / uptime,
                "decode_frames": metrics.decode_frames / uptime,
                "decode_tokens": metrics.decode_tokens / uptime,
            },
            "counts": {
                "turn_detection_calls": metrics.td_calls,
                "pre_encode_calls": metrics.pre_encode_calls,
                "encode_calls": metrics.encode_calls,
                "decode_calls": metrics.decode_calls,
            },
        }

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self._metrics = RunnerMetrics(start_time=time.monotonic())

    def queue_turn_detection(self, seq: Sequence) -> None:
        self._td_ready_queue.put(seq)

    def queue_pre_encode(self, seq: Sequence) -> None:
        self._queue_seq(seq, "pre_encode")

    def queue_encode(self, seq: Sequence) -> None:
        self._queue_seq(seq, "encode")

    def queue_decode(self, seq: Sequence) -> None:
        self._queue_seq(seq, "decode")

    def _queue_seq(self, seq: Sequence, kind: str) -> None:
        if kind == "pre_encode":
            queue_obj = self._pre_encode_ready_queue
            flag_attr = "pre_encode_queued"
        elif kind == "encode":
            queue_obj = self._encode_ready_queue
            flag_attr = "encode_queued"
        elif kind == "decode":
            queue_obj = self._decode_ready_queue
            flag_attr = "decode_queued"
        else:
            raise ValueError(f"Unknown queue kind: {kind}")

        with seq.lock:
            if seq.status == SequenceStatus.FINISHED:
                return
            if seq.status != SequenceStatus.RUNNING:
                if kind == "pre_encode":
                    return
                if kind in ("encode", "decode"):
                    return
            if getattr(seq, flag_attr):
                return
            if kind == "pre_encode" and not seq.raw_queue:
                return
            if kind == "encode" and not seq.has_chunk_ready():
                return
            if kind == "decode" and not seq.has_encoded():
                return
            setattr(seq, flag_attr, True)
        queue_obj.put(seq)
        with self._queue_depth_lock:
            self._queue_depths[kind] += 1

    def _pop_ready(self, kind: str) -> Sequence | None:
        if kind == "pre_encode":
            queue_obj = self._pre_encode_ready_queue
            flag_attr = "pre_encode_queued"
        elif kind == "encode":
            queue_obj = self._encode_ready_queue
            flag_attr = "encode_queued"
        elif kind == "decode":
            queue_obj = self._decode_ready_queue
            flag_attr = "decode_queued"
        else:
            raise ValueError(f"Unknown queue kind: {kind}")
        try:
            seq = queue_obj.get_nowait()
        except queue.Empty:
            return None
        with self._queue_depth_lock:
            self._queue_depths[kind] = max(0, self._queue_depths[kind] - 1)
        with seq.lock:
            setattr(seq, flag_attr, False)
        return seq

    def _turn_detection_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            self._turn_detection_step()
            self._stop_event.wait(interval)

    def _encode_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            self._encode_step()
            self._stop_event.wait(interval)

    def _decode_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            results = self._decode_step()
            result_ids = set()
            for result in results:
                self._results_queue.put(result)
                result_ids.add(result.seq_id)
            finalized = self._finalize_sequences(result_ids)
            if results or finalized:
                self._notify_update()
            self._stop_event.wait(interval)

    def _turn_detection_step(self) -> None:
        start = time.perf_counter()
        with record_function("turn_detection_step"):
            self._turn_detection_step_inner(start)

    def _turn_detection_step_inner(self, start: float) -> None:
        ready_seqs: list[Sequence] = []
        while True:
            try:
                ready_seqs.append(self._td_ready_queue.get_nowait())
            except queue.Empty:
                break

        if not ready_seqs:
            return

        seq_chunks: list[tuple[Sequence, np.ndarray]] = []
        for seq in ready_seqs:
            if seq.status == SequenceStatus.FINISHED:
                with seq.lock:
                    seq.td_queued = False
                continue
            try:
                chunk = seq.td_queue.get_nowait()
            except queue.Empty:
                with seq.lock:
                    seq.last_state = None
                    seq.td_queued = False
                continue
            seq_chunks.append((seq, chunk))

        if seq_chunks:
            self.turn_detection.process_batch(seq_chunks)

        turn_updates = 0
        for seq, _ in seq_chunks:
            with seq.lock:
                turn_position = seq.turn_position
                if turn_position and turn_position != seq.last_emitted_turn_position:
                    seq.last_emitted_turn_position = turn_position
                    self._results_queue.put(DecodeResult(seq.request_id, [], []))
                    turn_updates += 1

        for seq, _ in seq_chunks:
            requeue = False
            with seq.lock:
                if not seq.td_queue.empty():
                    requeue = True
                else:
                    seq.td_queued = False
            if requeue:
                self._td_ready_queue.put(seq)

        if turn_updates:
            self._notify_update()

        elapsed = time.perf_counter() - start
        if seq_chunks:
            with self._metrics_lock:
                self._metrics.td_time += elapsed
                self._metrics.td_calls += 1
                self._metrics.td_sequences += len(seq_chunks)

    def _pre_encode_step(self) -> None:
        admitted = self.scheduler.admit_ready()
        for seq in admitted:
            self.queue_pre_encode(seq)

        max_batch = max(1, int(getattr(self.config, "pre_encode_batch_size", 32)))

        while True:
            items: list[tuple[Sequence, np.ndarray, bool]] = []
            meta: list[tuple[Sequence, torch.Tensor | None, int, int]] = []
            while len(items) < max_batch:
                seq = self._pop_ready("pre_encode")
                if seq is None:
                    break
                requeue = False
                with seq.lock:
                    if seq.status == SequenceStatus.FINISHED:
                        continue
                    if not seq.raw_queue:
                        continue
                    samples, is_final = seq.raw_queue.popleft()
                    pre_encode_cache = seq.pre_encode_cache
                    pre_encode_cache_size = seq.pre_encode_cache_size
                    drop_extra_pre_encoded = seq.drop_extra_pre_encoded
                    if seq.raw_queue:
                        requeue = True
                if requeue:
                    self.queue_pre_encode(seq)
                items.append((seq, samples, is_final))
                meta.append(
                    (
                        seq,
                        pre_encode_cache,
                        pre_encode_cache_size,
                        drop_extra_pre_encoded,
                    )
                )

            if not items:
                return

            start = time.perf_counter()
            with record_function("pre_encode_step"), torch.inference_mode():
                feats_list = self._feature_extractor.batch_push(
                    items, device=self.device
                )
                # Ensure feature extractor output matches model dtype.
                if self._amp_dtype is not None:
                    feats_list = [
                        feats.to(dtype=self._amp_dtype) if feats is not None else None
                        for feats in feats_list
                    ]

                feats_in_list: list[torch.Tensor] = []
                feats_meta: list[tuple[Sequence, torch.Tensor, bool, int, int]] = []
                for (
                    (seq, _, _),
                    (
                        _,
                        pre_encode_cache,
                        pre_encode_cache_size,
                        drop_extra_pre_encoded,
                    ),
                    feats,
                ) in zip(items, meta, feats_list):
                    if feats is None:
                        continue
                    if pre_encode_cache is None:
                        feats_in = feats
                        has_cache = False
                    else:
                        cache_len = pre_encode_cache.size(-1)
                        feat_len = feats.size(-1)
                        feats_in = torch.empty(
                            feats.size(0),
                            feats.size(1),
                            cache_len + feat_len,
                            device=feats.device,
                            dtype=feats.dtype,
                        )
                        feats_in[..., :cache_len].copy_(pre_encode_cache)
                        feats_in[..., cache_len:].copy_(feats)
                        has_cache = True
                    feats_in_list.append(feats_in)
                    feats_meta.append(
                        (
                            seq,
                            feats_in,
                            has_cache,
                            pre_encode_cache_size,
                            drop_extra_pre_encoded,
                        )
                    )

                if not feats_in_list:
                    elapsed = time.perf_counter() - start
                    with self._metrics_lock:
                        self._metrics.pre_encode_time += elapsed
                        self._metrics.pre_encode_calls += 1
                        self._metrics.pre_encode_chunks += len(items)
                    continue

                max_len = max(feats_in.size(-1) for feats_in in feats_in_list)
                batch_size = len(feats_in_list)
                feat_dim = feats_in_list[0].size(1)
                self._shape_logger.log(
                    "pre_encode_in",
                    batch=batch_size,
                    feat_dim=feat_dim,
                    max_len=max_len,
                    dtype=str(feats_in_list[0].dtype),
                    device=str(feats_in_list[0].device),
                )
                batch_feats = torch.zeros(
                    (batch_size, feat_dim, max_len),
                    device=self.device,
                    dtype=feats_in_list[0].dtype,
                )
                lengths = []
                for idx, feats_in in enumerate(feats_in_list):
                    length = feats_in.size(-1)
                    batch_feats[idx : idx + 1, :, :length].copy_(feats_in)
                    lengths.append(length)
                length_tensor = torch.tensor(
                    lengths, device=self.device, dtype=torch.int64
                )
                if self.device.type == "cuda":
                    autocast_ctx = torch.autocast(
                        device_type="cuda",
                        dtype=self._amp_dtype,
                        enabled=self._amp_enabled,
                    )
                else:
                    autocast_ctx = torch.autocast(device_type="cpu", enabled=False)
                with autocast_ctx:
                    pre_encoded, out_lengths = self.pre_encoder(
                        batch_feats.transpose(1, 2), length_tensor
                    )
                self._shape_logger.log(
                    "pre_encode_out",
                    batch=pre_encoded.size(0),
                    time=pre_encoded.size(1),
                    hidden=pre_encoded.size(2),
                    len_min=int(out_lengths.min().item()),
                    len_max=int(out_lengths.max().item()),
                    dtype=str(pre_encoded.dtype),
                    device=str(pre_encoded.device),
                )

            for idx, (
                seq,
                feats_in,
                has_cache,
                pre_encode_cache_size,
                drop_extra_pre_encoded,
            ) in enumerate(feats_meta):
                out_len = int(out_lengths[idx].item())
                if out_len <= 0:
                    continue
                out = pre_encoded[idx : idx + 1, :out_len, :]
                if has_cache and drop_extra_pre_encoded > 0:
                    if out.size(1) <= drop_extra_pre_encoded:
                        continue
                    out = out[:, drop_extra_pre_encoded:, :]
                should_queue = False
                with seq.lock:
                    if seq.enc_buffer is None:
                        continue
                    if pre_encode_cache_size > 0:
                        if feats_in.size(-1) >= pre_encode_cache_size:
                            seq.pre_encode_cache = feats_in[
                                :, :, -pre_encode_cache_size:
                            ].detach()
                        else:
                            seq.pre_encode_cache = feats_in.detach()
                    if out.numel() > 0:
                        if seq.enc_buffer.numel() == 0:
                            seq.enc_buffer = out
                        else:
                            old_len = seq.enc_buffer.size(1)
                            new_len = out.size(1)
                            new_buffer = torch.empty(
                                (
                                    seq.enc_buffer.size(0),
                                    old_len + new_len,
                                    out.size(2),
                                ),
                                device=out.device,
                                dtype=out.dtype,
                            )
                            new_buffer[:, :old_len, :].copy_(seq.enc_buffer)
                            new_buffer[:, old_len:, :].copy_(out)
                            seq.enc_buffer = new_buffer
                    if seq.has_chunk_ready():
                        should_queue = True
                if should_queue:
                    self.queue_encode(seq)

            elapsed = time.perf_counter() - start
            with self._metrics_lock:
                self._metrics.pre_encode_time += elapsed
                self._metrics.pre_encode_calls += 1
                self._metrics.pre_encode_chunks += len(items)

    def _encode_step(self) -> None:
        self._pre_encode_step()

        start = time.perf_counter()
        with record_function("encode_step"):
            chunks: list[torch.Tensor] = []
            lengths: list[int] = []
            encode_seqs: list[Sequence] = []
            max_batch = max(1, int(getattr(self.config, "encode_batch_size", 32)))
            while len(encode_seqs) < max_batch:
                seq = self._pop_ready("encode")
                if seq is None:
                    break
                requeue = False
                with seq.lock:
                    if seq.status == SequenceStatus.FINISHED:
                        continue
                    if not seq.has_chunk_ready():
                        continue
                    chunk, length = seq.pop_chunk()
                    if chunk is None:
                        continue
                    seq.in_flight += 1
                    if seq.has_chunk_ready():
                        requeue = True
                if requeue:
                    self.queue_encode(seq)
                chunks.append(chunk)
                lengths.append(int(length.item()))
                encode_seqs.append(seq)

            if not chunks:
                return

            max_len = max(lengths)
            batch_size = len(chunks)
            hidden = chunks[0].size(2)
            graph_entry = self._encode_graphs.get(batch_size)
            use_graph = (
                graph_entry is not None
                and graph_entry.batch.size(1) == max_len
                and graph_entry.batch.size(2) == hidden
            )
            if use_graph:
                graph_entry.batch.zero_()
                for idx, (chunk, length) in enumerate(zip(chunks, lengths)):
                    graph_entry.batch[idx : idx + 1, :length, :].copy_(chunk)
                graph_entry.lengths.copy_(
                    torch.tensor(lengths, device=self.device, dtype=torch.int64)
                )
                self.scheduler.pack_states_into(
                    encode_seqs, graph_entry.batch_state, graph_entry.slot_index
                )
                pad_mask, att_mask = self.encoder._create_streaming_masks(
                    current_len=max_len,
                    cache_lengths=graph_entry.batch_state.cache_lengths,
                    processed_frames=graph_entry.batch_state.processed_frames,
                    device=self.device,
                    lengths=graph_entry.lengths,
                )
                graph_entry.pad_mask.copy_(pad_mask)
                graph_entry.attn_mask.copy_(att_mask)
                graph_entry.graph.replay()

                chunk_out = graph_entry.out
                batch_state = graph_entry.batch_state
                slot_index = graph_entry.slot_index
            else:
                batch = torch.zeros(
                    (batch_size, max_len, hidden),
                    device=chunks[0].device,
                    dtype=chunks[0].dtype,
                )
                for idx, (chunk, length) in enumerate(zip(chunks, lengths)):
                    batch[idx : idx + 1, :length, :].copy_(chunk)
                length_tensor = torch.tensor(
                    lengths, device=self.device, dtype=torch.int64
                )
                batch_state, slot_index = self.scheduler.pack_states(encode_seqs)
                if batch_state is not None:
                    cache_lengths = batch_state.cache_lengths
                    cache_min = int(cache_lengths.min().item())
                    cache_max = int(cache_lengths.max().item())
                else:
                    cache_min = 0
                    cache_max = 0
                self._shape_logger.log(
                    "encode_in",
                    batch=batch_size,
                    max_len=max_len,
                    hidden=hidden,
                    len_min=min(lengths),
                    len_max=max(lengths),
                    cache_min=cache_min,
                    cache_max=cache_max,
                    att_left=self.encoder.att_context_size[0],
                    att_right=self.encoder.att_context_size[1],
                    key_len_max=self.encoder.att_context_size[0] + max_len,
                    dtype=str(batch.dtype),
                    device=str(batch.device),
                )

                with torch.inference_mode():
                    if self.device.type == "cuda":
                        autocast_ctx = torch.autocast(
                            device_type="cuda",
                            dtype=self._amp_dtype,
                            enabled=self._amp_enabled,
                        )
                    else:
                        autocast_ctx = torch.autocast(
                            device_type="cpu", enabled=False
                        )
                    with autocast_ctx:
                        chunk_out, batch_state = self.encoder(
                            batch,
                            batch_state,
                            length=length_tensor,
                            bypass_pre_encode=True,
                        )
                self._shape_logger.log(
                    "encode_out",
                    batch=chunk_out.size(0),
                    time=chunk_out.size(1),
                    hidden=chunk_out.size(2),
                    dtype=str(chunk_out.dtype),
                    device=str(chunk_out.device),
                )

            for idx, seq in enumerate(encode_seqs):
                out = chunk_out[idx : idx + 1, :, : lengths[idx]]
                should_decode = False
                with seq.lock:
                    seq.enqueue_encoded(out)
                    if seq.has_encoded():
                        should_decode = True
                    seq.in_flight = max(0, seq.in_flight - 1)
                if should_decode:
                    self.queue_decode(seq)

            self.scheduler.unpack_states(batch_state, slot_index)

            elapsed = time.perf_counter() - start
            with self._metrics_lock:
                self._metrics.encode_time += elapsed
                self._metrics.encode_calls += 1
                self._metrics.encode_frames += sum(lengths)

    def _decode_step(self) -> list[DecodeResult]:
        start = time.perf_counter()
        with record_function("decode_step"):
            frames = 0
            results: list[DecodeResult] = []
            batch_seqs: list[Sequence] = []
            batch_chunks: list[torch.Tensor] = []
            max_batch = max(1, int(getattr(self.config, "decode_batch_size", 32)))
            while len(batch_seqs) < max_batch:
                seq = self._pop_ready("decode")
                if seq is None:
                    break
                requeue = False
                with seq.lock:
                    if seq.status == SequenceStatus.FINISHED:
                        continue
                    if not seq.has_encoded():
                        continue
                    chunk = seq.pop_encoded()
                    if chunk is None:
                        continue
                    if seq.has_encoded():
                        requeue = True
                if requeue:
                    self.queue_decode(seq)
                batch_seqs.append(seq)
                batch_chunks.append(chunk)
                frames += int(chunk.size(2))

            if batch_seqs:
                token_lists, confidence_scores = self._decode_batch(
                    batch_seqs, batch_chunks
                )
                for seq, tokens, scores in zip(
                    batch_seqs, token_lists, confidence_scores
                ):
                    if tokens:
                        with seq.lock:
                            seq.append_tokens(tokens, scores)
                        results.append(DecodeResult(seq.request_id, tokens, scores))

            if batch_seqs:
                total_tokens = sum(len(tokens) for tokens in token_lists)
                elapsed = time.perf_counter() - start
                with self._metrics_lock:
                    self._metrics.decode_time += elapsed
                    self._metrics.decode_calls += 1
                    self._metrics.decode_frames += frames
                    self._metrics.decode_tokens += total_tokens

            return results

    def _finalize_sequences(self, result_ids: set[int]) -> bool:
        finalized = False
        for seq in self.scheduler.active_sequences():
            if seq.status == SequenceStatus.FINISHED:
                continue
            with seq.lock:
                if not seq.final:
                    continue
                if seq.raw_queue or seq.encoded_queue:
                    continue
                if seq.enc_buffer is None or seq.enc_buffer.numel() > 0:
                    continue
                if seq.in_flight > 0:
                    continue
            self._release_sequence(seq)
            if seq.request_id not in result_ids:
                self._results_queue.put(DecodeResult(seq.request_id, [], []))
            finalized = True
        return finalized

    def _decode_batch(self, seqs: list[Sequence], chunks: list[torch.Tensor]):
        lengths = [int(chunk.size(2)) for chunk in chunks]
        order = sorted(range(len(seqs)), key=lengths.__getitem__, reverse=True)
        sorted_seqs = [seqs[i] for i in order]
        sorted_chunks = [chunks[i] for i in order]
        sorted_lengths = [lengths[i] for i in order]
        sorted_pred_outs = [seq.pred_out for seq in sorted_seqs]
        sorted_pred_states = [seq.pred_state for seq in sorted_seqs]
        token_lists_sorted = [[] for _ in sorted_seqs]
        confidence_scores_sorted = [[] for _ in sorted_seqs]

        with torch.inference_mode():
            max_len = sorted_lengths[0]
            batch_size = len(sorted_seqs)
            frame_hidden = sorted_chunks[0].size(1)
            chunk_batch = torch.zeros(
                (batch_size, frame_hidden, max_len),
                device=sorted_chunks[0].device,
                dtype=self._dtype,
            )
            for idx, chunk in enumerate(sorted_chunks):
                length = sorted_lengths[idx]
                chunk_batch[idx, :, :length].copy_(
                    chunk.to(dtype=self._dtype).squeeze(0)
                )

            pred_out_batch = torch.cat(
                [pred.to(dtype=self._dtype) for pred in sorted_pred_outs], dim=0
            )
            state_batch = self._stack_states(sorted_pred_states)

            active = batch_size
            for t in range(max_len):
                while active > 0 and t >= sorted_lengths[active - 1]:
                    active -= 1
                if active == 0:
                    break

                frame_batch = chunk_batch[:active, :, t].unsqueeze(1)
                if isinstance(state_batch, tuple):
                    state_active = tuple(part[:, :active, :] for part in state_batch)
                else:
                    state_active = state_batch[:, :active, :]

                pred_out_active, state_active, new_tokens, confidence_scores = (
                    self._decode_frame_batch(
                        frame_batch,
                        pred_out_batch[:active],
                        state_active,
                    )
                )

                pred_out_batch[:active].copy_(pred_out_active)
                if isinstance(state_batch, tuple):
                    for idx, part in enumerate(state_batch):
                        part[:, :active, :].copy_(state_active[idx])
                else:
                    state_batch[:, :active, :].copy_(state_active)

                for row, (tokens, conf_score) in enumerate(
                    zip(new_tokens, confidence_scores)
                ):
                    if tokens:
                        token_lists_sorted[row].extend(tokens)
                        confidence_scores_sorted[row].extend(conf_score)

        split_pred_outs = torch.split(pred_out_batch, 1, dim=0)
        split_states = self._unstack_states(state_batch)
        for seq, pred_out, pred_state in zip(
            sorted_seqs, split_pred_outs, split_states
        ):
            seq.pred_out = pred_out
            seq.pred_state = pred_state

        token_lists = [[] for _ in seqs]
        confidence_scores = [[] for _ in seqs]
        for sorted_idx, original_idx in enumerate(order):
            token_lists[original_idx] = token_lists_sorted[sorted_idx]
            confidence_scores[original_idx] = confidence_scores_sorted[sorted_idx]
        return token_lists, confidence_scores

    def _decode_frame_batch(
        self,
        frame_batch: torch.Tensor,
        pred_out_batch: torch.Tensor,
        state_batch,
    ):
        token_lists = [[] for _ in range(frame_batch.size(0))]
        token_confs = [[] for _ in range(frame_batch.size(0))]

        for _ in range(self._max_symbols_per_frame):
            logits = self.joiner(frame_batch, pred_out_batch)
            ids = logits.argmax(-1)
            blank_mask = ids == self.blank_id
            if blank_mask.all().item():
                break

            log_probs = torch.log_softmax(logits, dim=-1)
            token_logp = log_probs.gather(-1, ids.unsqueeze(1)).squeeze(-1)
            token_conf = token_logp.exp()

            non_blank_idx = (~blank_mask).nonzero(as_tuple=False).flatten()
            if non_blank_idx.numel() > 0:
                ids_nb = ids.index_select(0, non_blank_idx).detach().cpu().tolist()
                conf_nb = (
                    token_conf.index_select(0, non_blank_idx).detach().cpu().tolist()
                )
                rows = non_blank_idx.detach().cpu().tolist()
                for row, token_id, conf in zip(rows, ids_nb, conf_nb):
                    token_lists[row].append(token_id)
                    token_confs[row].append(conf)

            new_pred_out, new_state = self.decoder(ids, state=state_batch)
            pred_out_batch = torch.where(
                blank_mask.unsqueeze(-1), pred_out_batch, new_pred_out
            )

            state_batch = tuple(
                torch.where(blank_mask.view(1, -1, 1), old, new)
                for old, new in zip(state_batch, new_state)
            )

        return pred_out_batch, state_batch, token_lists, token_confs

    def _stack_states(self, states):
        if isinstance(states[0], tuple):
            num_parts = len(states[0])
            stacked = []
            for idx in range(num_parts):
                stacked.append(torch.cat([st[idx] for st in states], dim=1))
            return tuple(stacked)
        return torch.cat(states, dim=1)

    def _unstack_states(self, state_batch):
        if isinstance(state_batch, tuple):
            batch_size = state_batch[0].size(1)
            split_parts = [torch.split(part, 1, dim=1) for part in state_batch]
            outputs = []
            for i in range(batch_size):
                outputs.append(
                    tuple(split_parts[idx][i].detach() for idx in range(len(split_parts)))
                )
            return outputs
        batch_size = state_batch.size(1)
        return [chunk.detach() for chunk in torch.split(state_batch, 1, dim=1)]

    def _release_sequence(self, seq: Sequence) -> None:
        self.scheduler.release(seq)
        if seq.encoder_state is not None:
            self.scheduler.release_state_slot(seq.encoder_state)
            seq.encoder_state = None
        seq.cleanup()

    def _notify_update(self) -> None:
        with self._update_cond:
            self._update_seq += 1
            self._update_cond.notify_all()
