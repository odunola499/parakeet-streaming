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
        self._window_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._fb_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._pinned_batch: torch.Tensor | None = None
        self._pinned_shape = (0, 0)
        self._device_batch: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _get_seq_len(self, seq_len: int) -> int:
        pad_amount = self.n_fft
        return (seq_len + pad_amount - self.n_fft) // self.hop_length

    def _get_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        window = self._window_cache.get(key)
        if window is None:
            window = self.window.to(device=device, dtype=dtype)
            self._window_cache[key] = window
        return window

    def _get_fb(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        fb = self._fb_cache.get(key)
        if fb is None:
            fb = self.fb.to(device=device, dtype=dtype)
            self._fb_cache[key] = fb
        return fb

    def _compute_features_tensor(
        self, x: torch.Tensor, *, center: bool = True
    ) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dtype != torch.float32:
            x = x.float()

        x = x + torch.randn_like(x) * self.dither
        right = x[:, 1:] - self.preemph * x[:, :-1]
        x = torch.cat((x[:, :1], right), dim=-1)

        window = self._get_window(x.device, torch.float32)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=center,
            return_complex=True,
            pad_mode="constant",
        )

        spec = torch.view_as_real(stft)
        spec = torch.sqrt(spec.pow(2).sum(-1))
        spec = spec.pow(self.mag_power)

        fb = self._get_fb(spec.device, spec.dtype)
        mel = torch.matmul(fb, spec)
        mel = torch.log(mel + self.log_zero_guard_value)

        return mel

    def _compute_features(self, waveform: np.ndarray | torch.Tensor):
        if isinstance(waveform, np.ndarray):
            x = torch.from_numpy(waveform)
        else:
            x = waveform
        return self._compute_features_tensor(x)

    def _ensure_ring_capacity(self, state, extra: int) -> None:
        if extra <= 0:
            return
        needed = state._fe_buf_len + extra
        if state._fe_buf_capacity >= needed:
            return
        new_capacity = max(needed, 1024, state._fe_buf_capacity * 2)
        new_buffer = np.empty((new_capacity,), dtype=np.float32)
        if state._fe_buf_len:
            start = state._fe_buf_start
            end = start + state._fe_buf_len
            if end <= state._fe_buf_capacity:
                new_buffer[: state._fe_buf_len] = state._fe_buffer[start:end]
            else:
                first = state._fe_buf_capacity - start
                new_buffer[:first] = state._fe_buffer[start:]
                remaining = state._fe_buf_len - first
                if remaining:
                    new_buffer[first : first + remaining] = state._fe_buffer[:remaining]
        state._fe_buffer = new_buffer
        state._fe_buf_capacity = new_capacity
        state._fe_buf_start = 0

    def _append_samples(self, state, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self._ensure_ring_capacity(state, samples.size)
        write_pos = (state._fe_buf_start + state._fe_buf_len) % state._fe_buf_capacity
        end_pos = write_pos + samples.size
        if end_pos <= state._fe_buf_capacity:
            state._fe_buffer[write_pos:end_pos] = samples
        else:
            first = state._fe_buf_capacity - write_pos
            state._fe_buffer[write_pos:] = samples[:first]
            remaining = samples.size - first
            if remaining:
                state._fe_buffer[:remaining] = samples[first:]
        state._fe_buf_len += samples.size

    def _copy_from_ring_numpy(
        self,
        state,
        start_rel: int,
        length: int,
        dest: np.ndarray,
        dest_offset: int,
    ) -> None:
        if length <= 0:
            return
        if state._fe_buf_capacity == 0:
            return
        start = (state._fe_buf_start + start_rel) % state._fe_buf_capacity
        end = start + length
        if end <= state._fe_buf_capacity:
            dest[dest_offset : dest_offset + length] = state._fe_buffer[start:end]
            return
        first = state._fe_buf_capacity - start
        dest[dest_offset : dest_offset + first] = state._fe_buffer[start:]
        remaining = length - first
        if remaining:
            dest[dest_offset + first : dest_offset + length] = state._fe_buffer[
                :remaining
            ]

    def _copy_from_ring_tensor(
        self,
        state,
        start_rel: int,
        length: int,
        dest: torch.Tensor,
        dest_offset: int,
    ) -> None:
        if length <= 0:
            return
        if state._fe_buf_capacity == 0:
            return
        start = (state._fe_buf_start + start_rel) % state._fe_buf_capacity
        end = start + length
        if end <= state._fe_buf_capacity:
            dest[dest_offset : dest_offset + length].copy_(
                torch.from_numpy(state._fe_buffer[start:end])
            )
            return
        first = state._fe_buf_capacity - start
        dest[dest_offset : dest_offset + first].copy_(
            torch.from_numpy(state._fe_buffer[start:])
        )
        remaining = length - first
        if remaining:
            dest[dest_offset + first : dest_offset + length].copy_(
                torch.from_numpy(state._fe_buffer[:remaining])
            )

    def _build_segment(
        self,
        state,
        start_frame: int,
        max_frame: int,
    ) -> tuple[np.ndarray, int] | None:
        hop = self.hop_length
        pad = self.n_fft // 2
        seg_start = start_frame * hop - pad
        seg_end = max_frame * hop + pad
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            return None
        offset = state._fe_sample_offset
        buf_len = state._fe_buf_len
        left_pad = max(0, offset - seg_start)
        right_pad = max(0, seg_end - (offset + buf_len))
        avail_len = seg_len - left_pad - right_pad
        source_start_rel = max(0, seg_start - offset)
        segment = np.zeros((seg_len,), dtype=np.float32)
        if avail_len > 0:
            self._copy_from_ring_numpy(
                state, source_start_rel, avail_len, segment, left_pad
            )
        return segment, max_frame - start_frame + 1

    def __call__(self, audio, return_tensors: str | None = None, padding: bool = True):
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1

        features = self._compute_features(audio)
        return features

    def push(self, samples: np.ndarray, state, final: bool = False):
        if samples.size:
            self._append_samples(state, samples)
        else:
            return None

        total_samples = state._fe_sample_offset + state._fe_buf_len
        pad = self.n_fft // 2
        offset_frames = state._fe_sample_offset // self.hop_length

        if final:
            max_frame = total_samples // self.hop_length
        else:
            if total_samples <= pad:
                return None
            max_frame = (total_samples - pad) // self.hop_length

        if state._fe_sample_offset == 0:
            min_frame = 0
        else:
            min_frame = (
                state._fe_sample_offset + pad + self.hop_length - 1
            ) // self.hop_length

        start_frame = max(state._fe_emitted_frames, min_frame)
        if start_frame > max_frame:
            return None

        segment_info = self._build_segment(state, start_frame, max_frame)
        if segment_info is None:
            return None
        segment, frame_count = segment_info
        if segment.size == 0:
            return None
        device = getattr(state, "device", torch.device("cpu"))
        segment_tensor = torch.from_numpy(segment).to(device=device)
        features = self._compute_features_tensor(segment_tensor, center=False)
        new_feats = features[:, :, :frame_count]
        state._fe_emitted_frames = max_frame + 1
        self._trim_samples(state)
        return new_feats

    def batch_push(self, items, device: torch.device):
        if not items:
            return []

        hop = self.hop_length
        pad = self.n_fft // 2
        results = [None for _ in items]
        meta: list[tuple[int, object, int, int, int, int, int, int]] = []
        seg_lens: list[int] = []

        for idx, (state, samples, final) in enumerate(items):
            if samples.size:
                self._append_samples(state, samples)
            else:
                results[idx] = None
                continue

            total_samples = state._fe_sample_offset + state._fe_buf_len
            if final:
                max_frame = total_samples // hop
            else:
                if total_samples <= pad:
                    results[idx] = None
                    continue
                max_frame = (total_samples - pad) // hop

            if state._fe_sample_offset == 0:
                min_frame = 0
            else:
                min_frame = (state._fe_sample_offset + pad + hop - 1) // hop

            start_frame = max(state._fe_emitted_frames, min_frame)
            if start_frame > max_frame:
                results[idx] = None
                continue

            offset = state._fe_sample_offset
            buf_len = state._fe_buf_len
            seg_start = start_frame * hop - pad
            seg_end = max_frame * hop + pad
            seg_len = seg_end - seg_start
            left_pad = max(0, offset - seg_start)
            right_pad = max(0, seg_end - (offset + buf_len))
            avail_len = seg_len - left_pad - right_pad
            source_start_rel = max(0, seg_start - offset)
            frame_count = max_frame - start_frame + 1
            meta.append(
                (
                    idx,
                    state,
                    frame_count,
                    source_start_rel,
                    avail_len,
                    left_pad,
                    seg_len,
                    max_frame,
                )
            )
            seg_lens.append(seg_len)

        if not meta:
            return results

        batch_size = len(meta)
        max_len = max(seg_lens)
        if device.type == "cuda":
            if (
                self._pinned_batch is None
                or self._pinned_shape[0] < batch_size
                or self._pinned_shape[1] < max_len
            ):
                self._pinned_batch = torch.empty(
                    (batch_size, max_len), dtype=torch.float32, pin_memory=True
                )
                self._pinned_shape = (batch_size, max_len)
            batch_host = self._pinned_batch[:batch_size, :max_len]
            batch_host.zero_()

            key = (device, torch.float32)
            dev_batch = self._device_batch.get(key)
            if dev_batch is None or dev_batch.size(0) < batch_size or dev_batch.size(1) < max_len:
                dev_batch = torch.empty(
                    (batch_size, max_len), device=device, dtype=torch.float32
                )
                self._device_batch[key] = dev_batch
            batch_device = dev_batch[:batch_size, :max_len]
            for row, (_, state, _, source_start_rel, avail_len, left_pad, _, _) in enumerate(
                meta
            ):
                if avail_len <= 0:
                    continue
                self._copy_from_ring_tensor(
                    state, source_start_rel, avail_len, batch_host[row], left_pad
                )
            batch_device.copy_(batch_host, non_blocking=True)
            batch = batch_device
        else:
            batch = torch.zeros((batch_size, max_len), device=device, dtype=torch.float32)
            for row, (_, state, _, source_start_rel, avail_len, left_pad, _, _) in enumerate(
                meta
            ):
                if avail_len <= 0:
                    continue
                self._copy_from_ring_tensor(
                    state, source_start_rel, avail_len, batch[row], left_pad
                )

        features = self._compute_features_tensor(batch, center=False)

        for batch_idx, (
            out_idx,
            state,
            frame_count,
            _source_start_rel,
            _avail_len,
            _left_pad,
            _seg_len,
            max_frame,
        ) in enumerate(meta):
            if frame_count <= 0:
                results[out_idx] = None
                continue
            new_feats = features[batch_idx : batch_idx + 1, :, :frame_count]
            results[out_idx] = new_feats
            state._fe_emitted_frames = max_frame + 1
            self._trim_samples(state)

        return results

    def _trim_samples(self, state) -> None:
        if state._fe_buf_len == 0 or state._fe_buf_capacity == 0:
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
        if drop >= state._fe_buf_len:
            state._fe_buf_start = 0
            state._fe_buf_len = 0
        else:
            state._fe_buf_start = (
                state._fe_buf_start + drop
            ) % state._fe_buf_capacity
            state._fe_buf_len -= drop
        state._fe_sample_offset = aligned
