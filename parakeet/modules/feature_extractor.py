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

        self.samples = np.empty((0,), dtype=np.float32)
        self.emitted_frames = 0
        self._sample_offset = 0

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

    def push(self, samples: np.ndarray, final: bool = False):
        if samples.size:
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32, copy=False)
            self.samples = np.concatenate([self.samples, samples])
        if self.samples.size == 0:
            return None

        total_samples = self._sample_offset + self.samples.size
        hop = self.hop_length
        pad = self.n_fft // 2
        offset_frames = self._sample_offset // hop

        if final:
            max_global = total_samples // hop
        else:
            if total_samples <= pad:
                return None
            max_global = (total_samples - pad) // hop

        if self._sample_offset == 0:
            min_global = 0
        else:
            min_global = (self._sample_offset + pad + hop - 1) // hop

        start_global = max(self.emitted_frames, min_global)
        if start_global > max_global:
            return None

        features = self._compute_features(self.samples)
        start_idx = start_global - offset_frames
        end_idx = max_global - offset_frames
        new_feats = features[:, :, start_idx : end_idx + 1]
        self.emitted_frames = max_global + 1
        self._trim_samples()
        return new_feats

    def _trim_samples(self) -> None:
        if self.samples.size == 0:
            return
        hop = self.hop_length
        pad = self.n_fft // 2
        min_keep = max(0, self.emitted_frames * hop - pad)
        if min_keep <= self._sample_offset:
            return
        aligned = (min_keep // hop) * hop
        if aligned <= self._sample_offset:
            return
        drop = aligned - self._sample_offset
        if drop >= self.samples.size:
            self.samples = np.empty((0,), dtype=np.float32)
        else:
            self.samples = self.samples[drop:]
        self._sample_offset = aligned
