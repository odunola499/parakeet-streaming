from collections import deque
import threading
from typing import Dict

import numpy as np
import torch

from parakeet.config import Config
from parakeet.engine.model_runner import ModelRunner
from parakeet.engine.scheduler import Scheduler, StreamResult
from parakeet.engine.sequence import Sequence


class ASREngine:

    def __init__(self, config: Config, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.scheduler = Scheduler(max_active=config.max_num_streams)
        self.runner = ModelRunner(config, self.device, self.scheduler)
        self.tokenizer = self.runner.model._tokenizer
        self.streams: Dict[int, Sequence] = {}
        self._stream_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._stream_results: Dict[int, deque[StreamResult]] = {}

        self.runner.start_workers()

    def close(self) -> None:
        self.runner.stop_workers()

    def create_stream(self) -> int:
        seq = self.runner.create_sequence()
        if seq is None:
            raise RuntimeError("No free streaming state available.")
        self.runner.add_sequence(seq)
        with self._stream_lock:
            self.streams[seq.request_id] = seq
        with self._result_lock:
            self._stream_results[seq.request_id] = deque()
        return seq.request_id

    def push_samples(
        self, stream_id: int, samples: np.ndarray, final: bool = False
    ) -> None:
        with self._stream_lock:
            seq = self.streams.get(stream_id)
        if seq is None:
            return
        with seq.lock:
            seq.push_samples(samples, final=final)

    def get_update_seq(self) -> int:
        return self.runner.get_update_seq()

    def wait_for_update(self, last_seq: int, timeout: float | None = None) -> int:
        return self.runner.wait_for_update(last_seq, timeout=timeout)

    def _drain_to_stream_results_locked(self) -> None:
        items = self.runner.drain_results()
        for item in items:
            with self._stream_lock:
                seq = self.streams.get(item.seq_id)
            if seq is None:
                continue
            with seq.lock:
                text = self.tokenizer.decode(list(seq.token_ids))
                is_final = seq.is_finished
            stream_result = StreamResult(
                stream_id=item.seq_id,
                text=text,
                token_ids=item.token_ids,
                is_final=is_final,
                confidence_scores=item.confidence_scores,
            )
            self._stream_results.setdefault(item.seq_id, deque()).append(stream_result)
            if is_final:
                self.cleanup_stream(item.seq_id)

    def collect_stream_results(self, stream_id: int) -> list[StreamResult]:
        with self._result_lock:
            self._drain_to_stream_results_locked()
            queue = self._stream_results.get(stream_id)
            if not queue:
                return []
            results = list(queue)
            queue.clear()
            return results

    def cleanup_stream(self, stream_id: int) -> None:
        with self._stream_lock:
            seq = self.streams.pop(stream_id, None)
        if seq is None:
            return

    def get_stream(self, stream_id: int) -> Sequence | None:
        with self._stream_lock:
            return self.streams.get(stream_id)

    def drop_stream_results(self, stream_id: int) -> None:
        with self._result_lock:
            self._stream_results.pop(stream_id, None)
