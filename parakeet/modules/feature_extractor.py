import math

import librosa
import numpy as np
import torch


class FeatureExtractor:
    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate=16000,
        n_mels=128,
        n_fft=512,
        window_size=0.025,
        window_stride=0.01,
        normalize="NA",
        preemph=0.97,
        dither=1e-5,
        log_zero_guard_value=2**-24,
        lowfreq=0,
        highfreq=None,
        mag_power=2.0,
        pad_value=0.0,
        mel_norm="slaney",
    ):
        super().__init__()

        self.sampling_rate = sampling_rate
        self.feature_size = n_mels
        self.padding_value = pad_value
        self.normalize = normalize
        self.n_mels = n_mels
        self.n_fft = n_fft or 2 ** math.ceil(
            math.log2(int(window_size * sampling_rate))
        )
        self.win_length = int(window_size * sampling_rate)
        self.hop_length = int(window_stride * sampling_rate)
        self.preemph = preemph
        self.dither = dither
        self.log_zero_guard_value = float(log_zero_guard_value)
        self.mag_power = mag_power
        self.pad_value = pad_value

        self.window = torch.hann_window(self.win_length, periodic=False)

        highfreq = highfreq or sampling_rate / 2
        fb = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=self.n_fft,
            n_mels=n_mels,
            fmin=lowfreq,
            fmax=highfreq,
            norm=mel_norm,
        ).astype(np.float32)
        self.fb = torch.from_numpy(fb)

    def _get_seq_len(self, seq_len: int) -> int:
        pad_amount = self.n_fft
        return (seq_len + pad_amount - self.n_fft) // self.hop_length

    def _compute_features(self, waveform: np.ndarray):
        x = torch.from_numpy(waveform).unsqueeze(0)

        x = x + torch.randn_like(x) * self.dither
        right = x[:, 1:] - self.preemph * x[:, :-1]
        x = torch.cat((x[:, :1], right), dim=-1)

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(dtype=torch.float32),
            center=True,
            return_complex=True,
            pad_mode="constant",
        )

        spec = torch.view_as_real(stft)
        spec = torch.sqrt(spec.pow(2).sum(-1))
        spec = spec.pow(self.mag_power)

        mel = torch.matmul(self.fb.to(dtype=spec.dtype), spec)
        mel = torch.log(mel + self.log_zero_guard_value)

        return mel

    def __call__(self, audio, return_tensors: str | None = None, padding: bool = True):
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1

        features = self._compute_features(audio)
        return features

    def push(self, samples: np.ndarray, state, final: bool = False):
        if samples.size:
            if state._fe_samples.size:
                state._fe_samples = np.concatenate([state._fe_samples, samples])
            else:
                state._fe_samples = samples.copy()
        else:
            return None

        total_samples = state._fe_sample_offset + state._fe_samples.size
        hop = self.hop_length
        pad = self.n_fft // 2
        offset_frames = state._fe_sample_offset // hop

        if final:
            max_frame = total_samples // hop
        else:
            if total_samples <= pad:
                return None
            max_frame = (total_samples - pad) // hop

        if state._fe_sample_offset == 0:
            min_frame = 0
        else:
            min_frame = (state._fe_sample_offset + pad + hop - 1) // hop

        start_frame = max(state._fe_emitted_frames, min_frame)
        if start_frame > max_frame:
            return None

        features = self._compute_features(state._fe_samples)
        start_idx = start_frame - offset_frames
        end_idx = max_frame - offset_frames
        new_feats = features[:, :, start_idx : end_idx + 1]
        state._fe_emitted_frames = max_frame + 1
        self._trim_samples(state)
        return new_feats

    def _trim_samples(self, state) -> None:
        if state._fe_samples.size == 0:
            return
        hop = self.hop_length
        pad = self.n_fft // 2
        min_keep = max(0, state._fe_emitted_frames * hop - pad)
        if min_keep <= state._fe_sample_offset:
            return
        aligned = (min_keep // hop) * hop
        if aligned <= state._fe_sample_offset:
            return
        drop = aligned - state._fe_sample_offset
        if drop >= state._fe_samples.size:
            state._fe_samples = np.empty((0,), dtype=np.float32)
        else:
            state._fe_samples = state._fe_samples[drop:]
        state._fe_sample_offset = aligned
