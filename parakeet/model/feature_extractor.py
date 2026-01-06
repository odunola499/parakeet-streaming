import math

import librosa
import torch
from typing import List
import numpy as np


class ParakeetFeatureExtractor:
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
        log=True,
        log_zero_guard_value=2**-24,
        lowfreq=0,
        highfreq=None,
        mag_power=2.0,
        pad_value=0.0,
        mel_norm="slaney",
        **kwargs,
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
        self.log = log
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

    def _compute_features(self, waveform: List[np.ndarray], seq_len_time: int):
        x = torch.tensor(waveform, dtype=torch.float32)

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
        if self.mag_power != 1.0:
            spec = spec.pow(self.mag_power)

        mel = torch.matmul(self.fb.to(dtype=spec.dtype), spec)
        if self.log:
            mel = torch.log(mel + self.log_zero_guard_value)

        return mel

    def __call__(
        self,
        audio,
        return_tensors: str | None = None,
        padding: bool = True,
        pad_to: int = None,
    ):
        if isinstance(audio, np.ndarray):
            if audio.ndim == 1:
                audio_list = [audio]
            elif audio.ndim == 2:
                audio_list = [audio[i] for i in range(audio.shape[0])]
            else:
                raise ValueError("audio must be 1D or 2D numpy array")
        elif isinstance(audio, (list, tuple)):
            audio_list = list(audio)
        else:
            raise ValueError("audio must be a numpy array or a list of numpy arrays")

        features = self._compute_features(audio_list, audio_list[0].shape[-1])
        if pad_to:
            time_length = features.shape[-1]
            assert time_length <= pad_to, "time_length must be less than pad_to"
            remains = pad_to - time_length
            features = torch.nn.functional.pad(features, (0, remains))

        return {"input_features": features}


if __name__ == "__main__":
    import numpy as np

    feature_extractor = ParakeetFeatureExtractor()
    array = np.random.randn(2000)
    print(array.shape)
    print(feature_extractor([array] * 2)["input_features"].shape)
