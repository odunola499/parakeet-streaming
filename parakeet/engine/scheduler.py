from dataclasses import dataclass
from collections import deque
from typing import Deque, Iterable
import threading

import torch

from parakeet.modules.cache import StreamingState
from parakeet.engine.sequence import Sequence, SequenceStatus


@dataclass
class StreamResult:
    stream_id: int
    text: str
    token_ids: list[int]
    is_final: bool


class Scheduler:
    def __init__(self, max_active: int = 64):
        self.max_active = max_active
        self.waiting: Deque[Sequence] = deque()
        self.active: Deque[Sequence] = deque()
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pool_state: StreamingState | None = None
        self._batch_state: StreamingState | None = None
        self._batch_state_capacity = 0
        self._free_slots: Deque[int] = deque()
        self._state_encoder = None
        self._state_device: torch.device | None = None
        self._state_dtype: torch.dtype | None = None

    def add(self, seq: Sequence) -> None:
        with self._lock:
            if len(self.active) < self.max_active:
                seq.status = SequenceStatus.RUNNING
                self.active.append(seq)
            else:
                seq.status = SequenceStatus.WAITING
                self.waiting.append(seq)

    def admit_ready(self) -> list[Sequence]:
        admitted: list[Sequence] = []
        with self._lock:
            while self.waiting and len(self.active) < self.max_active:
                seq = self.waiting.popleft()
                if seq.status == SequenceStatus.FINISHED:
                    continue
                seq.status = SequenceStatus.RUNNING
                self.active.append(seq)
                admitted.append(seq)
        return admitted

    def release(self, seq: Sequence) -> None:
        with self._lock:
            if seq in self.waiting:
                self.waiting.remove(seq)
            if seq in self.active:
                self.active.remove(seq)
            seq.status = SequenceStatus.FINISHED

    def active_sequences(self) -> Iterable[Sequence]:
        with self._lock:
            return list(self.active)

    def idle(self) -> bool:
        with self._lock:
            return not self.waiting and not self.active

    def init_state_pool(
        self,
        encoder,
        device: torch.device,
        dtype: torch.dtype,
        pool_size: int,
    ) -> None:
        with self._state_lock:
            self._state_encoder = encoder
            self._state_device = device
            self._state_dtype = dtype
            self._pool_state = encoder.init_streaming_state(
                batch_size=pool_size, device=device, dtype=dtype
            )
            self._batch_state = None
            self._batch_state_capacity = 0
            self._free_slots = deque(range(pool_size))

    def _reset_state(self, slot: int) -> None:
        if self._pool_state is None:
            return
        self._pool_state.processed_frames[slot : slot + 1].zero_()
        self._pool_state.cache_lengths[slot : slot + 1].zero_()
        self._pool_state.cache.reset_slot(slot)

    def acquire_state_slot(self) -> int | None:
        with self._state_lock:
            if not self._free_slots:
                return None
            slot = self._free_slots.popleft()
        self._reset_state(slot)
        return slot

    def release_state_slot(self, slot: int) -> None:
        self._reset_state(slot)
        with self._state_lock:
            self._free_slots.append(slot)

    def _get_batch_state(self, batch_size: int) -> StreamingState:
        if (
            self._state_encoder is None
            or self._state_device is None
            or self._state_dtype is None
        ):
            raise RuntimeError("State pool not initialized.")
        if self._batch_state is None or self._batch_state_capacity < batch_size:
            self._batch_state = self._state_encoder.init_streaming_state(
                batch_size=batch_size,
                device=self._state_device,
                dtype=self._state_dtype,
            )
            self._batch_state_capacity = batch_size
        if batch_size == self._batch_state_capacity:
            return self._batch_state
        return StreamingState(
            cache=self._batch_state.cache.view(batch_size),
            processed_frames=self._batch_state.processed_frames[:batch_size],
            cache_lengths=self._batch_state.cache_lengths[:batch_size],
        )

    def pack_states(
        self, seqs: Iterable[Sequence]
    ) -> tuple[StreamingState, torch.Tensor]:
        if self._pool_state is None:
            raise RuntimeError("State pool not initialized.")
        seqs = list(seqs)
        batch_size = len(seqs)
        batch_state = self._get_batch_state(batch_size)
        slot_ids = [seq.encoder_state for seq in seqs]
        slot_index = torch.tensor(slot_ids, device=self._state_device, dtype=torch.long)

        torch.index_select(
            self._pool_state.processed_frames,
            0,
            slot_index,
            out=batch_state.processed_frames,
        )
        torch.index_select(
            self._pool_state.cache_lengths,
            0,
            slot_index,
            out=batch_state.cache_lengths,
        )
        torch.index_select(
            self._pool_state.cache.attn_k,
            0,
            slot_index,
            out=batch_state.cache.attn_k,
        )
        torch.index_select(
            self._pool_state.cache.attn_v,
            0,
            slot_index,
            out=batch_state.cache.attn_v,
        )
        torch.index_select(
            self._pool_state.cache.conv,
            0,
            slot_index,
            out=batch_state.cache.conv,
        )

        return batch_state, slot_index

    def unpack_states(
        self, batch_state: StreamingState, slot_index: torch.Tensor
    ) -> None:
        if self._pool_state is None:
            return
        self._pool_state.processed_frames.index_copy_(
            0,
            slot_index,
            batch_state.processed_frames,
        )
        self._pool_state.cache_lengths.index_copy_(
            0,
            slot_index,
            batch_state.cache_lengths,
        )
        self._pool_state.cache.attn_k.index_copy_(
            0,
            slot_index,
            batch_state.cache.attn_k,
        )
        self._pool_state.cache.attn_v.index_copy_(
            0,
            slot_index,
            batch_state.cache.attn_v,
        )
        self._pool_state.cache.conv.index_copy_(
            0,
            slot_index,
            batch_state.cache.conv,
        )
