from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable
import queue
import threading

import torch
from torch.nn import functional as F

from parakeet.config import Config
from parakeet.engine.scheduler import Scheduler
from parakeet.engine.sequence import Sequence, SequenceStatus
from parakeet.model import Parakeet


@dataclass
class DecodeResult:
    seq_id: int
    token_ids: list[int]


class ModelRunner:
    def __init__(self, config: Config, device: torch.device, scheduler: Scheduler):
        self.config = config
        self.device = device
        self.scheduler = scheduler

        self._init_model()
        self._init_state_pool(pool_size=config.max_num_streams)

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._results_queue: queue.Queue[DecodeResult] = queue.Queue()

    def _init_model(self) -> None:
        self.model = Parakeet.from_pretrained(self.config.size).to(self.device)
        self.model.eval()

        self.encoder = self.model.encoder
        self.pre_encoder = self.encoder.pre_encode
        self.decoder = self.model.predictor
        self.joiner = self.model.joiner
        self.sampler = self.model

        param = next(self.model.parameters())
        self._dtype = param.dtype

        self._attn_cache_len = self.encoder.att_context_size[0]
        self._layer_hidden = [layer.hidden_size for layer in self.encoder.layers]
        self._conv_cache_len = [
            layer.conv.depthwise_conv._max_cache_len for layer in self.encoder.layers
        ]
        self._subsampling_factor = self.model.config.subsampling_factor
        self._lookahead = self.encoder.att_context_size[1]
        self._chunk_frames_first = 1 + self._subsampling_factor * self._lookahead
        self._chunk_frames_next = (
            self._subsampling_factor + self._subsampling_factor * self._lookahead
        )
        self._feature_extractor_cls = type(self.model._feature_extractor)
        self._samples_per_frame = self._feature_extractor_cls().hop_length

    def _init_state_pool(self, pool_size: int) -> None:
        self._state_pool: list = []
        self._free_states: deque = deque()

        for _ in range(pool_size):
            state = self.encoder.init_streaming_state(batch_size=1, device=self.device)
            self._ensure_state_buffers(state)
            self._reset_state(state)
            self._state_pool.append(state)
            self._free_states.append(state)

    def _ensure_state_buffers(self, state) -> None:
        for layer_idx, hidden in enumerate(self._layer_hidden):
            attn_cache = state.cache.attn_caches[layer_idx]
            if attn_cache.k_cache is None:
                attn_cache.k_cache = torch.zeros(
                    (1, self._attn_cache_len, hidden),
                    device=self.device,
                    dtype=self._dtype,
                )
                attn_cache.v_cache = torch.zeros(
                    (1, self._attn_cache_len, hidden),
                    device=self.device,
                    dtype=self._dtype,
                )

            conv_cache = state.cache.conv_caches[layer_idx].get()
            if conv_cache is None:
                conv_cache = torch.zeros(
                    (1, hidden, self._conv_cache_len[layer_idx]),
                    device=self.device,
                    dtype=self._dtype,
                )
                state.cache.conv_caches[layer_idx].update(conv_cache)

    def _reset_state(self, state) -> None:
        if torch.is_tensor(state.processed_frames):
            state.processed_frames.zero_()
        else:
            state.processed_frames = 0

        if torch.is_tensor(state.cache_lengths):
            state.cache_lengths.zero_()
        else:
            state.cache_lengths = 0

        state.cache.current_max_len = 0

        for layer_idx, hidden in enumerate(self._layer_hidden):
            attn_cache = state.cache.attn_caches[layer_idx]
            if attn_cache.k_cache is None:
                attn_cache.k_cache = torch.zeros(
                    (1, self._attn_cache_len, hidden),
                    device=self.device,
                    dtype=self._dtype,
                )
            else:
                attn_cache.k_cache.zero_()

            if attn_cache.v_cache is None:
                attn_cache.v_cache = torch.zeros(
                    (1, self._attn_cache_len, hidden),
                    device=self.device,
                    dtype=self._dtype,
                )
            else:
                attn_cache.v_cache.zero_()

            conv_cache = state.cache.conv_caches[layer_idx].get()
            if conv_cache is None:
                conv_cache = torch.zeros(
                    (1, hidden, self._conv_cache_len[layer_idx]),
                    device=self.device,
                    dtype=self._dtype,
                )
                state.cache.conv_caches[layer_idx].update(conv_cache)
            else:
                conv_cache.zero_()

    def _acquire_state(self):
        if not self._free_states:
            return None
        state = self._free_states.popleft()
        self._reset_state(state)
        return state

    def _release_state(self, state) -> None:
        self._reset_state(state)
        self._free_states.append(state)

    def create_sequence(self) -> Sequence | None:
        encoder_state = self._acquire_state()
        if encoder_state is None:
            return None

        feature_extractor = self._feature_extractor_cls()

        pred_state = self.decoder.init_state(1)
        if isinstance(pred_state, tuple):
            pred_state = tuple(s.to(self.device) for s in pred_state)
        else:
            pred_state = pred_state.to(self.device)

        blank_id = self.model.blank_id
        start_ids = torch.full((1,), blank_id, dtype=torch.long, device=self.device)
        pred_out, pred_state = self.decoder.step(start_ids, state=pred_state)

        pre_encode_cache_size = self.encoder.pre_encode_cache_size
        if isinstance(pre_encode_cache_size, (list, tuple)):
            pre_encode_cache_size = pre_encode_cache_size[1]
        drop_extra_pre_encoded = self.encoder.drop_extra_pre_encoded
        enc_chunk_size = self.encoder.att_context_size[1] + 1

        max_pending_samples = int(
            self.config.max_stream_seconds * self.config.sample_rate
        )
        return Sequence(
            enc_hidden_dim=self.model.config.enc_hidden_dim,
            enc_chunk_size=enc_chunk_size,
            encoder_state=encoder_state,
            pred_state=pred_state,
            pred_out=pred_out,
            feature_extractor=feature_extractor,
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
        self.scheduler.admit_ready()
        self._pre_encode_step()
        self._encode_step()
        return self._decode_step()

    def start_workers(self, interval: float = 0.001) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        self._threads = [
            threading.Thread(
                target=self._pre_encode_loop, args=(interval,), daemon=True
            ),
            threading.Thread(target=self._encode_loop, args=(interval,), daemon=True),
            threading.Thread(target=self._decode_loop, args=(interval,), daemon=True),
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

    def _pre_encode_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            self.scheduler.admit_ready()
            self._pre_encode_step()
            self._stop_event.wait(interval)

    def _encode_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            self._encode_step()
            self._stop_event.wait(interval)

    def _decode_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            results = self._decode_step()
            for result in results:
                self._results_queue.put(result)
            self._stop_event.wait(interval)

    def _pre_encode_step(self) -> None:
        for seq in self.scheduler.active_sequences():
            if seq.status == SequenceStatus.FINISHED:
                continue
            if seq.has_pending_audio():
                self._run_pre_encode(seq)

    def _run_pre_encode(self, seq: Sequence) -> None:
        while True:
            with seq.lock:
                if not seq.raw_queue or seq.feature_extractor is None:
                    return
                samples, is_final = seq.raw_queue.popleft()
                pre_encode_cache = seq.pre_encode_cache
                drop_extra_pre_encoded = seq.drop_extra_pre_encoded
                pre_encode_cache_size = seq.pre_encode_cache_size

            feats = seq.feature_extractor.push(samples, final=is_final)
            if feats is None:
                continue
            feats = feats.to(self.device)

            if pre_encode_cache is None:
                feats_in = feats
                has_cache = False
            else:
                feats_in = torch.cat([pre_encode_cache, feats], dim=-1)
                has_cache = True

            lengths = torch.full(
                (feats_in.size(0),),
                feats_in.size(-1),
                dtype=torch.int64,
                device=self.device,
            )
            pre_encoded, _ = self.pre_encoder(feats_in.transpose(1, 2), lengths)
            if has_cache and drop_extra_pre_encoded > 0:
                pre_encoded = pre_encoded[:, drop_extra_pre_encoded:, :]

            with seq.lock:
                if seq.feature_extractor is None:
                    return
                if pre_encode_cache_size > 0:
                    if feats_in.size(-1) >= pre_encode_cache_size:
                        seq.pre_encode_cache = feats_in[
                            :, :, -pre_encode_cache_size:
                        ].detach()
                    else:
                        seq.pre_encode_cache = feats_in.detach()
                if pre_encoded is not None and pre_encoded.numel() > 0:
                    seq.enc_buffer = torch.cat([seq.enc_buffer, pre_encoded], dim=1)

    def _encode_step(self) -> None:
        chunks = []
        lengths = []
        active_seqs: list[Sequence] = []
        for seq in self.scheduler.active_sequences():
            with seq.lock:
                if not seq.has_chunk_ready():
                    continue
                chunk, length = seq.pop_chunk()
            if chunk is None:
                continue
            with seq.lock:
                seq.in_flight += 1
            chunks.append(chunk)
            lengths.append(int(length.item()))
            active_seqs.append(seq)

        if self.config.num_concurrent_requests > 0:
            chunks = chunks[: self.config.num_concurrent_requests]
            lengths = lengths[: self.config.num_concurrent_requests]
            active_seqs = active_seqs[: self.config.num_concurrent_requests]

        if not chunks:
            return

        max_len = max(lengths)
        padded_chunks = []
        for chunk, length in zip(chunks, lengths):
            if length < max_len:
                pad = max_len - length
                chunk = F.pad(chunk, (0, 0, 0, pad))
            padded_chunks.append(chunk)

        batch = torch.cat(padded_chunks, dim=0)
        length_tensor = torch.tensor(lengths, device=self.device, dtype=torch.int64)
        batch_state = self._pack_states(active_seqs)

        with torch.inference_mode():
            chunk_out, batch_state = self.encoder(
                batch,
                batch_state,
                length=length_tensor,
                bypass_pre_encode=True,
            )

        for idx, seq in enumerate(active_seqs):
            out = chunk_out[idx : idx + 1, :, : lengths[idx]]
            with seq.lock:
                seq.enqueue_encoded(out)

        self._unpack_states(batch_state, active_seqs)
        for seq in active_seqs:
            with seq.lock:
                seq.in_flight = max(0, seq.in_flight - 1)

    def _decode_step(self) -> list[DecodeResult]:
        results: list[DecodeResult] = []
        batch_seqs: list[Sequence] = []
        batch_chunks: list[torch.Tensor] = []
        for seq in list(self.scheduler.active_sequences()):
            with seq.lock:
                if seq.status == SequenceStatus.FINISHED:
                    if seq.in_flight == 0:
                        self._release_sequence(seq)
                    continue
                if seq.has_encoded():
                    chunk = seq.pop_encoded()
                    if chunk is not None:
                        batch_seqs.append(seq)
                        batch_chunks.append(chunk)

        if batch_seqs:
            token_lists = self._decode_batch(batch_seqs, batch_chunks)
            for seq, tokens in zip(batch_seqs, token_lists):
                if tokens:
                    with seq.lock:
                        seq.append_tokens(tokens)
                    results.append(DecodeResult(seq.request_id, tokens))

        for seq in list(self.scheduler.active_sequences()):
            with seq.lock:
                if (
                    seq.final
                    and not seq.has_pending_audio()
                    and not seq.has_chunk_ready()
                    and not seq.has_encoded()
                ):
                    seq.status = SequenceStatus.FINISHED
                if (
                    seq.is_finished
                    and not seq.has_pending_audio()
                    and seq.in_flight == 0
                ):
                    self._release_sequence(seq)
        return results

    def _decode_batch(
        self, seqs: list[Sequence], chunks: list[torch.Tensor]
    ) -> list[list[int]]:
        token_lists = [[] for _ in seqs]
        lengths = [int(chunk.size(2)) for chunk in chunks]
        pred_outs = [seq.pred_out for seq in seqs]
        pred_states = [seq.pred_state for seq in seqs]
        blank_id = self.model.blank_id

        with torch.inference_mode():
            max_len = max(lengths)
            for t in range(max_len):
                active = [i for i, length in enumerate(lengths) if t < length]
                if not active:
                    break
                frame_batch = torch.cat(
                    [chunks[i][:, :, t].unsqueeze(1) for i in active], dim=0
                )
                pred_out_batch = torch.cat([pred_outs[i] for i in active], dim=0)
                state_batch = self._stack_states([pred_states[i] for i in active])

                pred_out_batch, state_batch, new_tokens = self._decode_frame_batch(
                    frame_batch,
                    pred_out_batch,
                    state_batch,
                    blank_id=blank_id,
                )

                for idx, tokens in zip(active, new_tokens):
                    token_lists[idx].extend(tokens)

                split_pred_outs = torch.split(pred_out_batch, 1, dim=0)
                split_states = self._unstack_states(state_batch)
                for out, st, idx in zip(split_pred_outs, split_states, active):
                    pred_outs[idx] = out
                    pred_states[idx] = st

        for seq, pred_out, pred_state in zip(seqs, pred_outs, pred_states):
            seq.pred_out = pred_out
            seq.pred_state = pred_state

        return token_lists

    def _decode_frame_batch(
        self,
        frame_batch: torch.Tensor,
        pred_out_batch: torch.Tensor,
        state_batch,
        blank_id: int,
    ):
        token_lists = [[] for _ in range(frame_batch.size(0))]
        for _ in range(self.sampler.max_symbols_per_timestep):
            logits = self.joiner(frame_batch, pred_out_batch)
            ids = logits.argmax(-1)
            blank_mask = ids == blank_id
            if bool(blank_mask.all()):
                break

            for idx, token_id in enumerate(ids.tolist()):
                if token_id != blank_id:
                    token_lists[idx].append(token_id)

            new_pred_out, new_state = self.decoder.step(ids, state=state_batch)
            pred_out_batch = torch.where(
                blank_mask.unsqueeze(-1), pred_out_batch, new_pred_out
            )
            if isinstance(state_batch, tuple):
                state_batch = tuple(
                    torch.where(blank_mask.view(1, -1, 1), old, new)
                    for old, new in zip(state_batch, new_state)
                )
            else:
                state_batch = torch.where(
                    blank_mask.view(1, -1, 1), state_batch, new_state
                )

        return pred_out_batch, state_batch, token_lists

    def _stack_states(self, states):
        if isinstance(states[0], tuple):
            stacked = []
            for idx in range(len(states[0])):
                stacked.append(torch.cat([st[idx] for st in states], dim=1))
            return tuple(stacked)
        return torch.cat(states, dim=1)

    def _unstack_states(self, state_batch):
        if isinstance(state_batch, tuple):
            batch_size = state_batch[0].size(1)
            outputs = []
            for i in range(batch_size):
                outputs.append(tuple(s[:, i : i + 1, :].detach() for s in state_batch))
            return outputs
        batch_size = state_batch.size(1)
        return [state_batch[:, i : i + 1, :].detach() for i in range(batch_size)]

    def _release_sequence(self, seq: Sequence) -> None:
        self.scheduler.release(seq)
        if seq.encoder_state is not None:
            self._release_state(seq.encoder_state)
            seq.encoder_state = None
        seq.cleanup()

    def _pack_states(self, seqs: Iterable[Sequence]):
        seqs = list(seqs)
        batch_size = len(seqs)
        batch_state = self.encoder.init_streaming_state(
            batch_size=batch_size, device=self.device
        )

        processed_frames = torch.cat(
            [self._as_tensor(seq.encoder_state.processed_frames) for seq in seqs],
            dim=0,
        )
        cache_lengths = torch.cat(
            [self._as_tensor(seq.encoder_state.cache_lengths) for seq in seqs],
            dim=0,
        )
        batch_state.processed_frames = processed_frames
        batch_state.cache_lengths = cache_lengths

        num_layers = len(self.encoder.layers)
        for layer_idx in range(num_layers):
            hidden = self._layer_hidden[layer_idx]
            attn_k = []
            attn_v = []
            conv_c = []
            for seq in seqs:
                layer_cache = seq.encoder_state.cache.attn_caches[layer_idx]
                k_cache = layer_cache.k_cache
                v_cache = layer_cache.v_cache
                if k_cache is None:
                    k_cache = torch.zeros(
                        (1, self._attn_cache_len, hidden),
                        device=self.device,
                        dtype=self._dtype,
                    )
                    v_cache = torch.zeros(
                        (1, self._attn_cache_len, hidden),
                        device=self.device,
                        dtype=self._dtype,
                    )
                attn_k.append(k_cache)
                attn_v.append(v_cache)

                conv_cache = seq.encoder_state.cache.conv_caches[layer_idx].get()
                if conv_cache is None:
                    conv_cache = torch.zeros(
                        (1, hidden, self._conv_cache_len[layer_idx]),
                        device=self.device,
                        dtype=self._dtype,
                    )
                conv_c.append(conv_cache)

            batch_state.cache.attn_caches[layer_idx].k_cache = torch.cat(attn_k, dim=0)
            batch_state.cache.attn_caches[layer_idx].v_cache = torch.cat(attn_v, dim=0)
            batch_state.cache.conv_caches[layer_idx].cache = torch.cat(conv_c, dim=0)

        return batch_state

    def _unpack_states(self, batch_state, seqs: Iterable[Sequence]) -> None:
        seqs = list(seqs)
        for idx, seq in enumerate(seqs):
            seq.encoder_state.processed_frames = batch_state.processed_frames[
                idx : idx + 1
            ].detach()
            seq.encoder_state.cache_lengths = batch_state.cache_lengths[
                idx : idx + 1
            ].detach()
            for layer_idx in range(len(self.encoder.layers)):
                seq_cache = seq.encoder_state.cache.attn_caches[layer_idx]
                seq_cache.k_cache = (
                    batch_state.cache.attn_caches[layer_idx]
                    .k_cache[idx : idx + 1]
                    .detach()
                )
                seq_cache.v_cache = (
                    batch_state.cache.attn_caches[layer_idx]
                    .v_cache[idx : idx + 1]
                    .detach()
                )
                seq.encoder_state.cache.conv_caches[layer_idx].cache = (
                    batch_state.cache.conv_caches[layer_idx]
                    .cache[idx : idx + 1]
                    .detach()
                )

    def _as_tensor(self, value):
        if torch.is_tensor(value):
            return value.to(device=self.device, non_blocking=True)
        return torch.tensor(
            [value], device=self.device, dtype=torch.int64, pin_memory=True
        )
