import librosa
import numpy as np
import torch

from parakeet.model.modeling_parakeet import Parakeet

path = "/Users/odunolajenrola/Documents/GitHub/parakeet-streaming/test.mp3"
device_str = "cpu"
size = "small"
decode = True
max_seconds = 2


class StreamingFeatureExtractor:
    def __init__(self, extractor):
        self.extractor = extractor
        self.n_fft = extractor.n_fft
        self.hop_length = extractor.hop_length
        self.samples = np.empty((0,), dtype=np.float32)
        self.emitted_frames = 0

    def push(self, samples: np.ndarray, final: bool = False):
        if samples.size:
            self.samples = np.concatenate([self.samples, samples])
        features = self.extractor(self.samples, pad_to=None)["input_features"]
        if not torch.is_tensor(features):
            features = torch.as_tensor(features)
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


class PreEncodeStreamer:
    def __init__(self, encoder, cache_size: int, drop_extra: int, device: torch.device):
        self.encoder = encoder
        self.cache_size = cache_size
        self.drop_extra = drop_extra
        self.device = device
        self.cache = None

    def push(self, feats: torch.Tensor | None):
        if feats is None or feats.numel() == 0:
            return None
        feats = feats.to(self.device)
        if self.cache is None:
            feats_in = feats
            has_cache = False
        else:
            feats_in = torch.cat([self.cache, feats], dim=-1)
            has_cache = True
        lengths = torch.full(
            (feats_in.size(0),),
            feats_in.size(-1),
            dtype=torch.int64,
            device=self.device,
        )
        pre_encoded, _ = self.encoder.pre_encode(feats_in.transpose(1, 2), lengths)
        if has_cache and self.drop_extra > 0:
            pre_encoded = pre_encoded[:, self.drop_extra :, :]
        if self.cache_size > 0:
            if feats_in.size(-1) >= self.cache_size:
                self.cache = feats_in[:, :, -self.cache_size :].detach()
            else:
                self.cache = feats_in.detach()
        return pre_encoded


def main():
    device = torch.device(device_str)
    model = Parakeet.from_pretrained(size).to(device)
    model.eval()

    array, sr = librosa.load(path, sr=16000, duration=max_seconds)
    array = array.astype(np.float32, copy=False)

    feature_extractor = model.get_feature_extractor()
    predictor = model.get_predictor()
    tokenizer = model.get_tokenizer()
    blank_id = model.blank_id

    subsampling_factor = model.config.subsampling_factor
    lookahead = model.encoder.att_context_size[1]
    chunk_frames_first = 1 + subsampling_factor * lookahead
    chunk_frames_next = subsampling_factor + subsampling_factor * lookahead

    pre_encode_cache_size = model.encoder.pre_encode_cache_size
    if isinstance(pre_encode_cache_size, (list, tuple)):
        pre_encode_cache_size = pre_encode_cache_size[1]
    drop_extra_pre_encoded = model.encoder.drop_extra_pre_encoded

    feature_stream = StreamingFeatureExtractor(feature_extractor)
    pre_encode_stream = PreEncodeStreamer(
        model.encoder, pre_encode_cache_size, drop_extra_pre_encoded, device
    )

    state = model.encoder.init_streaming_state(device=device)
    enc_chunk_size = model.encoder.att_context_size[1] + 1
    enc_buffer = torch.empty((1, 0, model.config.enc_hidden_dim), device=device)

    token_chunks = []

    batch_size = 1
    pred_state = predictor.init_state(batch_size)
    if isinstance(pred_state, tuple):
        pred_state = tuple(s.to(device) for s in pred_state)
    else:
        pred_state = pred_state.to(device)
    start_ids = torch.full((batch_size,), blank_id, dtype=torch.long, device=device)
    pred_out, pred_state = predictor.step(start_ids, state=pred_state)

    def run_encoder(final: bool = False):
        nonlocal enc_buffer, state, pred_out, pred_state
        while enc_buffer.size(1) >= enc_chunk_size or (
            final and enc_buffer.size(1) > 0
        ):
            take = (
                enc_chunk_size
                if enc_buffer.size(1) >= enc_chunk_size
                else enc_buffer.size(1)
            )
            chunk = enc_buffer[:, :take, :]
            enc_buffer = enc_buffer[:, take:, :]
            length = torch.full(
                (chunk.size(0),), take, dtype=torch.int64, device=device
            )
            chunk_out, state = model.encoder.forward_streaming(
                chunk, state, length=length, bypass_pre_encode=True
            )
            if decode:
                for step in range(chunk_out.size(2)):
                    frame = chunk_out[:, :, step].unsqueeze(1)
                    hyp, pred_out, pred_state = model._greedy_decode_frame(
                        frame, pred_out, pred_state
                    )
                    if hyp.numel() > 0:
                        token_chunks.append(hyp)

    pos = 0
    first = True
    samples_per_frame = feature_extractor.hop_length

    while pos < array.shape[0]:
        chunk_frames = chunk_frames_first if first else chunk_frames_next
        first = False
        chunk_samples = chunk_frames * samples_per_frame
        end = min(array.shape[0], pos + chunk_samples)
        chunk = array[pos:end]
        pos = end

        feats = feature_stream.push(chunk, final=False)
        pre_encoded = pre_encode_stream.push(feats)
        if pre_encoded is not None:
            enc_buffer = torch.cat([enc_buffer, pre_encoded], dim=1)
            run_encoder(final=False)
        if decode and token_chunks:
            ids = torch.cat(token_chunks, dim=-1)
            text = tokenizer.decode(ids.to("cpu").tolist())
            print(text)
            print("")

    feats = feature_stream.push(np.empty((0,), dtype=np.float32), final=True)
    pre_encoded = pre_encode_stream.push(feats)
    if pre_encoded is not None:
        enc_buffer = torch.cat([enc_buffer, pre_encoded], dim=1)
    run_encoder(final=True)

    if decode and token_chunks:
        ids = torch.cat(token_chunks, dim=-1)
        text = tokenizer.decode(ids.to("cpu").tolist())
        print(text)


if __name__ == "__main__":
    main()
