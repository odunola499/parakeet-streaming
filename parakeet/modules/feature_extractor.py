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
            self.samples = np.concatenate([self.samples, samples])
        features = self._compute_features(self.samples)
        total_frames = features.shape[-1]
        if final:
            stable_frames = total_frames
        else:
            stable_frames = (len(self.samples) - self.n_fft // 2) // self.hop_length + 1
            stable_frames = max(0, min(stable_frames, total_frames))
        if stable_frames <= self.emitted_frames:
            return None
        new_feats = features[:, :, self.emitted_frames : stable_frames]
        self.emitted_frames = stable_frames
        return new_feats
