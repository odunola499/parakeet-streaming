from dataclasses import dataclass
import queue
import threading

import torch

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
        self.scheduler.init_state_pool(
            self.encoder,
            self.device,
            self._dtype,
            pool_size=config.max_num_streams,
        )

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._results_queue: queue.Queue[DecodeResult] = queue.Queue()
        self._update_cond = threading.Condition()
        self._update_seq = 0

    def _init_model(self) -> None:
        self.model = Parakeet.from_pretrained(self.config.size).to(self.device)
        self.model.eval()

        self.encoder = self.model.encoder
        self.pre_encoder = self.encoder.pre_encode
        self.decoder = self.model.predictor
        self.joiner = self.model.joiner

        param = next(self.model.parameters())
        self._dtype = param.dtype

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
        self.scheduler.admit_ready()
        self._pre_encode_step()
        self._encode_step()
        results, _ = self._decode_step()
        return results

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

    def get_update_seq(self) -> int:
        with self._update_cond:
            return self._update_seq

    def wait_for_update(self, last_seq: int, timeout: float | None = None) -> int:
        with self._update_cond:
            if self._update_seq != last_seq:
                return self._update_seq
            self._update_cond.wait(timeout=timeout)
            return self._update_seq

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
            results, state_changed = self._decode_step()
            result_ids = set()
            for result in results:
                self._results_queue.put(result)
                result_ids.add(result.seq_id)
            finalized = self._finalize_sequences(result_ids)
            if results or state_changed or finalized:
                self._notify_update()
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
                if not seq.raw_queue:
                    return
                samples, is_final = seq.raw_queue.popleft()
                pre_encode_cache = seq.pre_encode_cache
                drop_extra_pre_encoded = seq.drop_extra_pre_encoded
                pre_encode_cache_size = seq.pre_encode_cache_size

            feats = self._feature_extractor.push(samples, seq, final=is_final)
            if feats is None:
                continue
            feats = feats.to(self.device)

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
                if pre_encode_cache_size > 0:
                    if feats_in.size(-1) >= pre_encode_cache_size:
                        seq.pre_encode_cache = feats_in[
                            :, :, -pre_encode_cache_size:
                        ].detach()
                    else:
                        seq.pre_encode_cache = feats_in.detach()
                if pre_encoded is not None and pre_encoded.numel() > 0:
                    if seq.enc_buffer.numel() == 0:
                        seq.enc_buffer = pre_encoded
                    else:
                        old_len = seq.enc_buffer.size(1)
                        new_len = pre_encoded.size(1)
                        new_buffer = torch.empty(
                            (
                                seq.enc_buffer.size(0),
                                old_len + new_len,
                                pre_encoded.size(2),
                            ),
                            device=pre_encoded.device,
                            dtype=pre_encoded.dtype,
                        )
                        new_buffer[:, :old_len, :].copy_(seq.enc_buffer)
                        new_buffer[:, old_len:, :].copy_(pre_encoded)
                        seq.enc_buffer = new_buffer

    def _encode_step(self) -> None:
        chunks = []
        lengths = []
        active_seqs: list[Sequence] = []
        for seq in self.scheduler.active_sequences():
            with seq.lock:
                if seq.status == SequenceStatus.FINISHED:
                    continue
                if not seq.has_chunk_ready():
                    continue
                chunk, length = seq.pop_chunk()
                if chunk is None:
                    continue
                seq.in_flight += 1
            chunks.append(chunk)
            lengths.append(int(length.item()))
            active_seqs.append(seq)

        if not chunks:
            return

        max_len = max(lengths)
        batch_size = len(chunks)
        hidden = chunks[0].size(2)
        batch = torch.zeros(
            (batch_size, max_len, hidden),
            device=chunks[0].device,
            dtype=chunks[0].dtype,
        )
        for idx, (chunk, length) in enumerate(zip(chunks, lengths)):
            batch[idx : idx + 1, :length, :].copy_(chunk)
        length_tensor = torch.tensor(lengths, device=self.device, dtype=torch.int64)
        batch_state, slot_index = self.scheduler.pack_states(active_seqs)

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

        self.scheduler.unpack_states(batch_state, slot_index)
        for seq in active_seqs:
            with seq.lock:
                seq.in_flight = max(0, seq.in_flight - 1)

    def _decode_step(self) -> tuple[list[DecodeResult], bool]:
        results: list[DecodeResult] = []
        state_changed = False
        batch_seqs: list[Sequence] = []
        batch_chunks: list[torch.Tensor] = []
        for seq in list(self.scheduler.active_sequences()):
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

        return results, state_changed

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
                self._results_queue.put(DecodeResult(seq.request_id, []))
            finalized = True
        return finalized

    def _decode_batch(
        self, seqs: list[Sequence], chunks: list[torch.Tensor]
    ) -> list[list[int]]:
        lengths = [int(chunk.size(2)) for chunk in chunks]
        order = sorted(range(len(seqs)), key=lengths.__getitem__, reverse=True)
        sorted_seqs = [seqs[i] for i in order]
        sorted_chunks = [chunks[i] for i in order]
        sorted_lengths = [lengths[i] for i in order]
        sorted_pred_outs = [seq.pred_out for seq in sorted_seqs]
        sorted_pred_states = [seq.pred_state for seq in sorted_seqs]
        token_lists_sorted = [[] for _ in sorted_seqs]

        with torch.inference_mode():
            max_len = sorted_lengths[0]
            batch_size = len(sorted_seqs)
            for t in range(max_len):
                if t == 0:
                    frame_hidden = sorted_chunks[0].size(1)
                    frame_batch = torch.empty(
                        (batch_size, 1, frame_hidden),
                        device=sorted_chunks[0].device,
                        dtype=sorted_chunks[0].dtype,
                    )
                    pred_proto = sorted_pred_outs[0]
                    pred_out_batch = torch.empty(
                        (batch_size,) + pred_proto.shape[1:],
                        device=pred_proto.device,
                        dtype=pred_proto.dtype,
                    )

                while batch_size > 0 and t >= sorted_lengths[batch_size - 1]:
                    batch_size -= 1
                if batch_size == 0:
                    break

                for row in range(batch_size):
                    frame = sorted_chunks[row][:, :, t].unsqueeze(1)
                    frame_batch[row : row + 1].copy_(frame)
                    pred_out_batch[row : row + 1].copy_(sorted_pred_outs[row])

                state_batch = self._stack_states(sorted_pred_states[:batch_size])
                pred_out_active, state_batch, new_tokens = self._decode_frame_batch(
                    frame_batch[:batch_size],
                    pred_out_batch[:batch_size],
                    state_batch,
                )

                for row, tokens in enumerate(new_tokens):
                    if tokens:
                        token_lists_sorted[row].extend(tokens)

                split_pred_outs = torch.split(pred_out_active, 1, dim=0)
                split_states = self._unstack_states(state_batch)
                for row in range(batch_size):
                    sorted_pred_outs[row] = split_pred_outs[row]
                    sorted_pred_states[row] = split_states[row]

        for seq, pred_out, pred_state in zip(
            sorted_seqs, sorted_pred_outs, sorted_pred_states
        ):
            seq.pred_out = pred_out
            seq.pred_state = pred_state

        token_lists = [[] for _ in seqs]
        for sorted_idx, original_idx in enumerate(order):
            token_lists[original_idx] = token_lists_sorted[sorted_idx]
        return token_lists

    def _decode_frame_batch(
        self,
        frame_batch: torch.Tensor,
        pred_out_batch: torch.Tensor,
        state_batch,
    ):
        token_lists = [[] for _ in range(frame_batch.size(0))]
        for _ in range(self._max_symbols_per_frame):
            logits = self.joiner(frame_batch, pred_out_batch)
            ids = logits.argmax(-1)
            blank_mask = ids == self.blank_id
            if bool(blank_mask.all()):
                break

            for idx, token_id in enumerate(ids.tolist()):
                if token_id != self.blank_id:
                    token_lists[idx].append(token_id)

            new_pred_out, new_state = self.decoder(ids, state=state_batch)
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
            num_parts = len(states[0])
            batch_size = len(states)
            for idx in range(num_parts):
                part0 = states[0][idx]
                out = torch.empty(
                    (part0.size(0), batch_size, part0.size(2)),
                    device=part0.device,
                    dtype=part0.dtype,
                )
                for b, st in enumerate(states):
                    out[:, b : b + 1, :].copy_(st[idx])
                stacked.append(out)
            return tuple(stacked)
        base = states[0]
        batch_size = len(states)
        out = torch.empty(
            (base.size(0), batch_size, base.size(2)),
            device=base.device,
            dtype=base.dtype,
        )
        for b, st in enumerate(states):
            out[:, b : b + 1, :].copy_(st)
        return out

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
            self.scheduler.release_state_slot(seq.encoder_state)
            seq.encoder_state = None
        seq.cleanup()

    def _notify_update(self) -> None:
        with self._update_cond:
            self._update_seq += 1
            self._update_cond.notify_all()
