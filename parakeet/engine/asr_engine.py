from collections import deque
import threading
from typing import Any, Dict

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
        notify_td = False
        with seq.lock:
            seq.push_samples(samples, final=final)
            if samples.size and not seq.td_queued:
                seq.td_queued = True
                notify_td = True
        if notify_td:
            self.runner.queue_turn_detection(seq)

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
                last_state = seq.last_state
                turn_detection = seq.turn_position
            stream_result = StreamResult(
                stream_id=item.seq_id,
                text=text,
                token_ids=item.token_ids,
                is_final=is_final,
                confidence_scores=item.confidence_scores,
                last_state=last_state,
                turn_detection=turn_detection,
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

    def get_metrics(self) -> dict[str, Any]:
        with self._stream_lock:
            streams = list(self.streams.values())
        connected = len(streams)
        raw_queue_depth = 0
        encoded_queue_depth = 0
        td_queue_depth = 0
        raw_queue_streams = 0
        encoded_queue_streams = 0
        td_queue_streams = 0
        in_flight = 0

        for seq in streams:
            with seq.lock:
                raw_len = len(seq.raw_queue)
                enc_len = len(seq.encoded_queue)
                td_len = seq.td_queue.qsize()
                if raw_len:
                    raw_queue_streams += 1
                if enc_len:
                    encoded_queue_streams += 1
                if td_len:
                    td_queue_streams += 1
                raw_queue_depth += raw_len
                encoded_queue_depth += enc_len
                td_queue_depth += td_len
                in_flight += seq.in_flight

        runner_metrics = self.runner.get_metrics()
        runner_metrics.update(
            {
                "connected_streams": connected,
                "queues": {
                    "raw_depth": raw_queue_depth,
                    "encoded_depth": encoded_queue_depth,
                    "turn_detection_depth": td_queue_depth,
                    "raw_streams": raw_queue_streams,
                    "encoded_streams": encoded_queue_streams,
                    "turn_detection_streams": td_queue_streams,
                },
                "in_flight": in_flight,
            }
        )
        return runner_metrics
