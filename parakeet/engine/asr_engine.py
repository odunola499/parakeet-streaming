from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch

from parakeet.config import Config
from parakeet.engine.model_runner import ModelRunner, DecodeResult
from parakeet.engine.scheduler import Scheduler, StreamResult
from parakeet.engine.sequence import Sequence


class ASREngine:

    def __init__(self, config: Config, device: str | torch.device = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.scheduler = Scheduler(max_active=config.max_num_streams)
        self.runner = ModelRunner(config, self.device, self.scheduler)
        self.tokenizer = self.runner.model.get_tokenizer()
        self.streams: Dict[int, Sequence] = {}

        self.runner.start_workers()

    def close(self) -> None:
        self.runner.stop_workers()

    def create_stream(self) -> int:
        seq = self.runner.create_sequence()
        if seq is None:
            raise RuntimeError("No free streaming state available.")
        self.runner.add_sequence(seq)
        self.streams[seq.request_id] = seq
        return seq.request_id

    def push_pcm(self, stream_id: int, pcm_bytes: bytes, final: bool = False) -> None:
        seq = self.streams[stream_id]
        with seq.lock:
            seq.push_pcm(pcm_bytes, final=final)

    def push_samples(
        self, stream_id: int, samples: np.ndarray, final: bool = False
    ) -> None:
        seq = self.streams[stream_id]
        with seq.lock:
            seq.push_samples(samples, final=final)

    def drain(self) -> list[DecodeResult]:
        return self.runner.drain_results()

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(token_ids))

    def collect_results(self) -> list[StreamResult]:
        results = []
        for item in self.drain():
            seq = self.streams.get(item.seq_id)
            if seq is None:
                continue
            text = self.decode(seq.token_ids)
            results.append(
                StreamResult(
                    stream_id=item.seq_id,
                    text=text,
                    token_ids=item.token_ids,
                    is_final=seq.is_finished,
                )
            )
            if seq.is_finished:
                self.cleanup_stream(item.seq_id)
        return results

    def cleanup_stream(self, stream_id: int) -> None:
        seq = self.streams.pop(stream_id, None)
        if seq is None:
            return
        with seq.lock:
            seq.cleanup()
