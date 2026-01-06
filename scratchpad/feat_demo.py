import librosa
import torch

from parakeet.model.config import ModelConfig
from parakeet.model.modeling_parakeet import Parakeet


def build_models(device: torch.device):
    offline = Parakeet.from_pretrained()
    offline.eval()
    offline.to(device)

    stream_cfg = ModelConfig()
    stream_cfg.stream = True
    streaming = Parakeet(stream_cfg)
    streaming.load_state_dict(offline.state_dict())
    streaming._feature_extractor = offline.get_feature_extractor()
    streaming._tokenizer = offline.get_tokenizer()
    streaming.eval()
    streaming.to(device)

    return offline, streaming


def main():
    path = "/Users/odunolajenrola/Documents/GitHub/parakeet-streaming/test.mp3"
    device_str = "cpu"
    chunk_size = 0
    check = True
    decode = True
    max_seconds = 100

    device = torch.device(device_str)
    offline, streaming = build_models(device)

    audio, _ = librosa.load(path, sr=16000, duration=max_seconds)
    features = offline.get_feature_extractor()(audio)["input_features"]
    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    features = features.to(device)
    lengths = torch.full(
        (features.size(0),), features.size(-1), dtype=torch.int64, device=device
    )

    with torch.no_grad():
        offline_enc = offline.encoder(features, lengths)
        pre_encoded, pre_len = offline.encoder.pre_encode(
            features.transpose(1, 2), lengths
        )

        state = streaming.encoder.init_streaming_state(
            batch_size=features.size(0), device=device
        )
        chunk_size = chunk_size or (streaming.encoder.att_context_size[1] + 1)

        stream_chunks = []
        if decode:
            predictor = streaming.get_predictor()
            tokenizer = streaming.get_tokenizer()
            blank_id = streaming.get_blank_id()
            batch_size = features.size(0)
            pred_state = predictor.init_state(batch_size)
            if isinstance(pred_state, tuple):
                pred_state = tuple(s.to(device) for s in pred_state)
            else:
                pred_state = pred_state.to(device)
            start_ids = torch.full(
                (batch_size,), blank_id, dtype=torch.long, device=device
            )
            pred_out, pred_state = predictor.step(start_ids, state=pred_state)
            token_chunks = []
        else:
            token_chunks = None

        for start in range(0, pre_encoded.size(1), chunk_size):
            chunk = pre_encoded[:, start : start + chunk_size, :]
            chunk_len = (pre_len - start).clamp(min=0, max=chunk_size)
            if (chunk_len == 0).all():
                break
            chunk_out, state = streaming.encoder.forward_streaming(
                chunk, state, length=chunk_len, bypass_pre_encode=True
            )
            stream_chunks.append(chunk_out)

            if token_chunks is not None:
                for t in range(chunk_out.size(2)):
                    frame = chunk_out[:, :, t].unsqueeze(1)
                    hyp, pred_out, pred_state = streaming._greedy_decode_frame(
                        frame, pred_out, pred_state
                    )
                    if hyp.numel() > 0:
                        token_chunks.append(hyp)

            if token_chunks:
                ids = torch.cat(token_chunks, dim=-1)
                text = tokenizer.decode(ids.to("cpu").tolist())
                print("streaming_text:", text)

        stream_enc = torch.cat(stream_chunks, dim=2)

        if check:
            max_diff = (offline_enc - stream_enc).abs().max().item()
            print(f"max_abs_diff: {max_diff:.6f}")

        if token_chunks is not None:
            if token_chunks:
                ids = torch.cat(token_chunks, dim=-1)
            else:
                ids = torch.empty(
                    (features.size(0), 0), dtype=torch.long, device=device
                )
            text = tokenizer.decode(ids.to("cpu").tolist())
            print("streaming_text:", text)


if __name__ == "__main__":
    main()
