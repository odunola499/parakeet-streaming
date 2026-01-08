from __future__ import annotations
from dataclasses import dataclass

from collections import deque
from typing import Deque, Iterable
import threading

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
