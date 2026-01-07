from itertools import count
from enum import Enum, auto
import numpy as np


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    sample_rate = 16_000
    hop_length = sample_rate * 0.01
    num_channels = 1
    n_fft = 512
    blank_id = 1026

    def __init__(self, feature_extractor):
        self.last_token = None
        self.request_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.incoming_bytes = bytearray()
        self.emitted_frames = 0
        self.audio_samples = np.empty(0, dtype=np.float32)
        self.feature_extractor = feature_extractor

        self.pred_state = None
        self.pre_encoded_frames = None
        self.encoded_frames = None

        self.num_tokens = 0
        self.block_table = []
        self.token_ids = []

    def __len__(self):
        return self.emitted_frames

    def __getitem__(self):
        pass

    def push(self, pcm_data: bytes):
        # how we get audio to this sequence
        # TODO: can clients tell server when audio is complete
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        self.audio_samples = np.concatenate([self.audio_samples, samples])
        features = self.feature_extractor(self.audio_samples, pad_to=None)[
            "input_features"
        ]
        features = features.squeeze(0)  # remove batched dim
        total_frames = features.shape[-1]
        new_features = features[:, self.emitted_frames : total_frames]
        self.emitted_frames += total_frames
        return new_features

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.emitted_frames,
        )
