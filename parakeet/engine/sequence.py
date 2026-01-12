from __future__ import annotations

from collections import deque
import threading
from enum import Enum, auto
from itertools import count

import numpy as np
import torch


# Add a SequenceStatus.PAUSE to show when audio isnt coming in,
# temporarily remove from worker batch batch to free memory
# Only a stream disconnect would assert a Finished, and this would free up space.


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(
        self,
        enc_hidden_dim: int,
        enc_chunk_size: int,
        encoder_state,
        pred_state,
        pred_out,
        pre_encode_cache_size: int,
        drop_extra_pre_encoded: int,
        chunk_samples_first: int,
        chunk_samples_next: int,
        max_pending_samples: int,
        device: torch.device,
    ):
        self.request_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.device = device
        self.lock = threading.Lock()

        self.encoder_state = encoder_state
        self.pred_state = pred_state
        self.pred_out = pred_out
        self.pre_encode_cache = None
        self.pre_encode_cache_size = pre_encode_cache_size
        self.drop_extra_pre_encoded = drop_extra_pre_encoded
        self.enc_chunk_size = enc_chunk_size
        self.enc_buffer = torch.empty((1, 0, enc_hidden_dim), device=device)
        self.raw_queue: deque[tuple[np.ndarray, bool]] = deque()
        self.encoded_queue: deque[torch.Tensor] = deque()
        self.chunk_samples_first = chunk_samples_first
        self.chunk_samples_next = chunk_samples_next
        self._pending_capacity = max_pending_samples
        self._pending_buffer = np.empty((self._pending_capacity,), dtype=np.float32)
        self._pending_start = 0
        self._pending_len = 0
        self._first_chunk = True
        self._fe_samples = np.empty((0,), dtype=np.float32)
        self._fe_emitted_frames = 0
        self._fe_sample_offset = 0

        self.token_ids: list[int] = []
        self.final = False
        self.in_flight = 0

    def push_pcm(self, pcm_bytes: bytes, final: bool = False):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        self.enqueue_samples(samples, final=final)

    def push_samples(self, samples: np.ndarray, final: bool = False):
        self.enqueue_samples(samples, final=final)

    def enqueue_samples(self, samples: np.ndarray, final: bool = False):
        if samples.size:
            if self._pending_len + samples.size > self._pending_capacity:
                raise RuntimeError("Stream buffer overflow.")
            write_pos = (
                self._pending_start + self._pending_len
            ) % self._pending_capacity
            end_pos = write_pos + samples.size
            if end_pos <= self._pending_capacity:
                self._pending_buffer[write_pos:end_pos] = samples
            else:
                first = self._pending_capacity - write_pos
                self._pending_buffer[write_pos:] = samples[:first]
                self._pending_buffer[: end_pos % self._pending_capacity] = samples[
                    first:
                ]
            self._pending_len += samples.size

        self._emit_chunks(final=final)
        if final:
            self.final = True

    def _pop_pending(self, num_samples: int) -> np.ndarray:
        if self._pending_len < num_samples:
            return np.empty((0,), dtype=np.float32)
        start = self._pending_start
        end = start + num_samples
        if end <= self._pending_capacity:
            chunk = self._pending_buffer[start:end].copy()
        else:
            chunk = np.concatenate(
                (
                    self._pending_buffer[start:],
                    self._pending_buffer[: end % self._pending_capacity],
                )
            ).copy()
        self._pending_start = end % self._pending_capacity
        self._pending_len -= num_samples
        return chunk

    def _emit_chunks(self, final: bool = False) -> None:
        while True:
            if self._first_chunk:
                needed = self.chunk_samples_first
            else:
                needed = self.chunk_samples_next
            if self._pending_len < needed:
                break
            chunk = self._pop_pending(needed)
            self.raw_queue.append((chunk, False))
            self._first_chunk = False
        if final and self._pending_len > 0:
            chunk = self._pop_pending(self._pending_len)
            self.raw_queue.append((chunk, True))

    def has_pending_audio(self) -> bool:
        return bool(self.raw_queue)

    def flush(self):
        self.enqueue_samples(np.empty((0,), dtype=np.float32), final=True)

    def has_chunk_ready(self) -> bool:
        return self.enc_buffer.size(1) >= self.enc_chunk_size or (
            self.final and self.enc_buffer.size(1) > 0
        )

    def pop_chunk(self):
        if not self.has_chunk_ready():
            return None, None
        take = (
            self.enc_chunk_size
            if self.enc_buffer.size(1) >= self.enc_chunk_size
            else self.enc_buffer.size(1)
        )
        chunk = self.enc_buffer[:, :take, :]
        self.enc_buffer = self.enc_buffer[:, take:, :]
        if self.final and take < self.enc_chunk_size:
            pad = self.enc_chunk_size - take
            pad_chunk = torch.zeros(
                (chunk.size(0), pad, chunk.size(2)),
                device=chunk.device,
                dtype=chunk.dtype,
            )
            chunk = torch.cat([chunk, pad_chunk], dim=1)
            take = self.enc_chunk_size
        length = torch.full((1,), take, dtype=torch.int64, device=self.device)
        return chunk, length

    def enqueue_encoded(self, chunk_out: torch.Tensor):
        if chunk_out is not None and chunk_out.numel() > 0:
            self.encoded_queue.append(chunk_out)

    def has_encoded(self) -> bool:
        return bool(self.encoded_queue)

    def pop_encoded(self) -> torch.Tensor | None:
        if not self.encoded_queue:
            return None
        return self.encoded_queue.popleft()

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def append_tokens(self, token_ids: list[int]) -> None:
        if token_ids:
            self.token_ids.extend(token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    def cleanup(self) -> None:
        self.raw_queue.clear()
        self.encoded_queue.clear()
        self.pre_encode_cache = None
        self.pred_state = None
        self.pred_out = None
        self.enc_buffer = None
        self._pending_buffer = None
        self._fe_samples = np.empty((0,), dtype=np.float32)
        self._fe_emitted_frames = 0
        self._fe_sample_offset = 0
