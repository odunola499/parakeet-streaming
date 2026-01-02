import math

import numpy as np
import librosa
import torch


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

    def _compute_features(self, waveform: np.ndarray, seq_len_time: int):
        x = torch.tensor(waveform, dtype=torch.float32)

        x = x + torch.randn_like(x) * self.dither
        x = torch.cat((x[:1], x[1:] - self.preemph * x[:-1]))

        if seq_len_time < x.shape[0]:
            timemask = torch.arange(x.shape[0]) < seq_len_time
            x = x.masked_fill(~timemask, 0.0)

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

        seq_len_frames = self._get_seq_len(seq_len_time)

        max_len = mel.shape[1]
        if seq_len_frames < max_len:
            mask = torch.arange(max_len) >= seq_len_frames
            mel = mel.masked_fill(mask.unsqueeze(0), self.pad_value)

        return mel

    def __call__(
        self,
        audio,
        lengths=None,
        return_tensors: str | None = None,
        padding: bool = True,
    ):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

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

        if lengths is None:
            lengths = [a.shape[-1] for a in audio_list]
        elif isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        else:
            lengths = list(lengths)

        if len(lengths) != len(audio_list):
            raise ValueError("lengths must match the number of audio samples")

        features = [
            self._compute_features(a if a.ndim == 1 else a.mean(axis=0), int(seq_len))
            for a, seq_len in zip(audio_list, lengths, strict=False)
        ]

        if padding or return_tensors is not None:
            max_len = max(f.shape[1] for f in features)
            padded = []
            for f in features:
                if f.shape[1] < max_len:
                    pad = max_len - f.shape[1]
                    f = torch.nn.functional.pad(f, (0, pad), value=self.pad_value)
                padded.append(f)
            features = padded

        if return_tensors is None:
            return {"input_features": [f.numpy().astype(np.float32) for f in features]}

        if return_tensors not in {"np", "pt"}:
            raise ValueError("return_tensors must be 'np' or 'pt'")

        if return_tensors == "np":
            stacked = np.stack([f.numpy().astype(np.float32) for f in features], axis=0)
            return {"input_features": stacked}

        stacked = torch.stack(features, dim=0)
        return {"input_features": stacked}


if __name__ == "__main__":
    import numpy as np
    feature_extractor = ParakeetFeatureExtractor()
    array = np.random.randn(4, 32000)
    print(array.shape)
    print(feature_extractor(array))
