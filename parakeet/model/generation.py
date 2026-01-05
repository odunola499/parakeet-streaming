from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union

import torch
from torch import Tensor


@dataclass
class Hypothesis:
    tokens: Tensor
    pred_out: Tensor
    pred_state: Tensor
    score: Tensor


@dataclass
class GenerationOutput:
    ids: Tensor
    labels: Union[List[str], List[int]]


class GenerationMixin(ABC):
    @abstractmethod
    def get_feature_extractor(self):
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self):
        raise NotImplementedError

    def get_encoder(self):
        if hasattr(self, "encoder"):
            return self.encoder
        raise NotImplementedError

    def get_joiner(self):
        if hasattr(self, "joiner"):
            return self.joiner
        raise NotImplementedError

    def get_predictor(self):
        if hasattr(self, "predictor"):
            return self.predictor
        raise NotImplementedError

    def get_sampler_type(self) -> str:
        if hasattr(self, "sampler_type"):
            return self.sampler_type
        config = getattr(self, "config", None)
        if config is not None and hasattr(config, "sampler_type"):
            return config.sampler_type
        return "greedy_search"

    def get_blank_id(self) -> int:
        return getattr(self, "blank_id", 0)

    def get_max_symbols_per_timestep(self) -> int:
        return getattr(self, "max_symbols_per_timestep", 5)

    def get_beam_size(self) -> int:
        return getattr(self, "beam_size", 4)

    def offline_decode(self, audio: Tensor) -> GenerationOutput:
        sampler_type = self.get_sampler_type()
        if sampler_type == "greedy_search":
            return self._greedy_decode(audio)
        if sampler_type == "beam_search":
            return self._beam_decode(audio)
        raise ValueError(f"Unknown sampler_type: {sampler_type}")

    def decode_features(self, features: Tensor) -> GenerationOutput:
        sampler_type = self.get_sampler_type()
        if sampler_type == "greedy_search":
            return self._greedy_decode_features(features)
        if sampler_type == "beam_search":
            return self._beam_decode_features(features)
        raise ValueError(f"Unknown sampler_type: {sampler_type}")

    def _greedy_decode(self, audio: Tensor) -> GenerationOutput:
        feature_extractor = self.get_feature_extractor()
        encoder = self.get_encoder()

        features = feature_extractor(audio)
        if isinstance(features, dict):
            if "input_values" in features:
                features = features["input_values"]
            elif "input_features" in features:
                features = features["input_features"]
            else:
                raise KeyError("feature_extractor output missing input values")
        else:
            if hasattr(features, "input_values"):
                features = features.input_values
            elif hasattr(features, "input_features"):
                features = features.input_features
            else:
                raise AttributeError("feature_extractor output missing input values")
        if not torch.is_tensor(features):
            features = torch.as_tensor(features)
        features = features.to(next(encoder.parameters()).device)
        encoder_output = encoder(features)
        return self._greedy_decode_from_encoder_output(encoder_output)

    def _greedy_decode_features(self, features: Tensor) -> GenerationOutput:
        encoder = self.get_encoder()
        encoder_output = encoder(features)
        return self._greedy_decode_from_encoder_output(encoder_output)

    def _greedy_decode_from_encoder_output(
        self, encoder_output: Tensor
    ) -> GenerationOutput:
        tokenizer = self.get_tokenizer()
        predictor = self.get_predictor()
        joiner = self.get_joiner()

        if encoder_output.dim() == 3:
            encoder_dim = getattr(joiner, "encoder_dim", None)
            if encoder_dim is not None:
                if encoder_output.shape[-1] == encoder_dim:
                    pass
                elif encoder_output.shape[1] == encoder_dim:
                    encoder_output = encoder_output.transpose(1, 2)
                else:
                    raise ValueError("encoder_output does not match encoder_dim")
        batch_size, frame_size = encoder_output.shape[:-1]
        pred_state = predictor.init_state(batch_size)
        if isinstance(pred_state, tuple):
            pred_state = tuple(state.to(encoder_output.device) for state in pred_state)
        else:
            pred_state = pred_state.to(encoder_output.device)
        blank_id = self.get_blank_id()
        start_ids = torch.full(
            (batch_size,), blank_id, dtype=torch.long, device=encoder_output.device
        )
        pred_out, pred_state = predictor.step(start_ids, state=pred_state)

        t = 0
        results = []
        while t < frame_size:
            frame = encoder_output[:, t : t + 1, :]
            hyp, pred_out, pred_state = self._greedy_decode_frame(
                frame, pred_out, pred_state
            )
            t += 1
            results.append(hyp)
        results = torch.concat(results, dim=-1)
        texts = tokenizer.decode(results.to("cpu").tolist())

        return GenerationOutput(ids=results, labels=texts)

    def _greedy_decode_frame(
        self,
        frame: Tensor,
        pred_out: Tensor,
        pred_state: Tensor,
    ):
        predictor = self.get_predictor()
        joiner = self.get_joiner()

        if frame.dim() == 2:
            frame = frame.unsqueeze(1)

        max_symbols = self.get_max_symbols_per_timestep()
        blank_id = self.get_blank_id()
        hyp = []
        while max_symbols > 0:
            max_symbols -= 1
            logits = joiner(frame, pred_out)
            ids = logits.argmax(-1)
            ids = ids.reshape(ids.shape[0], -1)
            ids = ids[:, -1]
            if (ids == blank_id).all():
                break

            hyp.append(ids.unsqueeze(-1))

            pred_out, pred_state = predictor.step(ids, state=pred_state)

        if hyp:
            hyp = torch.concat(hyp, dim=-1)
        else:
            hyp = torch.empty(
                (frame.shape[0], 0), dtype=torch.long, device=frame.device
            )
        return hyp, pred_out, pred_state

    def _beam_decode(self, audio: Tensor) -> GenerationOutput:
        feature_extractor = self.get_feature_extractor()
        encoder = self.get_encoder()

        features = feature_extractor(audio)
        if isinstance(features, dict):
            if "input_values" in features:
                features = features["input_values"]
            elif "input_features" in features:
                features = features["input_features"]
            else:
                raise KeyError("feature_extractor output missing input values")
        else:
            if hasattr(features, "input_values"):
                features = features.input_values
            elif hasattr(features, "input_features"):
                features = features.input_features
            else:
                raise AttributeError("feature_extractor output missing input values")
        if not torch.is_tensor(features):
            features = torch.as_tensor(features)
        features = features.to(next(encoder.parameters()).device)
        encoder_output = encoder(features)
        return self._beam_decode_from_encoder_output(encoder_output)

    def _beam_decode_features(self, features: Tensor) -> GenerationOutput:
        encoder = self.get_encoder()
        encoder_output = encoder(features)
        return self._beam_decode_from_encoder_output(encoder_output)

    def _beam_decode_from_encoder_output(
        self, encoder_output: Tensor
    ) -> GenerationOutput:
        tokenizer = self.get_tokenizer()
        predictor = self.get_predictor()
        joiner = self.get_joiner()

        if encoder_output.dim() == 3:
            encoder_dim = getattr(joiner, "encoder_dim", None)
            if encoder_dim is not None:
                if encoder_output.shape[-1] == encoder_dim:
                    pass
                elif encoder_output.shape[1] == encoder_dim:
                    encoder_output = encoder_output.transpose(1, 2)
                else:
                    raise ValueError("encoder_output does not match encoder_dim")
        batch_size, frame_size = encoder_output.shape[:-1]
        blank_id = self.get_blank_id()
        start_ids = torch.full(
            (batch_size, 1),
            fill_value=blank_id,
            dtype=torch.long,
            device=encoder_output.device,
        )

        pred_state = predictor.init_state(batch_size)
        if isinstance(pred_state, tuple):
            pred_state = tuple(state.to(encoder_output.device) for state in pred_state)
        else:
            pred_state = pred_state.to(encoder_output.device)
        pred_out, pred_state = predictor.step(start_ids, state=pred_state)

        hyps = [
            Hypothesis(
                tokens=torch.empty(
                    (batch_size, 0), dtype=torch.long, device=encoder_output.device
                ),
                pred_out=pred_out,
                pred_state=pred_state,
                score=torch.zeros(batch_size, 1, device=encoder_output.device),
            )
        ]

        for t in range(frame_size):
            frame = encoder_output[:, t : t + 1, :]
            hyps = self._beam_decode_frame(frame, hyps)

        best = hyps[0]
        texts = tokenizer.decode(best.tokens)
        return GenerationOutput(ids=best.tokens, labels=texts)

    def _beam_decode_frame(self, frame: Tensor, hyps: List[Hypothesis]):
        predictor = self.get_predictor()
        joiner = self.get_joiner()

        if frame.ndim == 2:
            frame = frame.unsqueeze(1)

        blank_id = self.get_blank_id()
        beam_size = self.get_beam_size()
        new_hyps = []
        for hyp in hyps:
            logits = joiner(frame, hyp.pred_out)
            logp = torch.log_softmax(logits, dim=-1)

            blank_score = logp[..., blank_id].squeeze(-1).squeeze(-1)
            hyp.score = hyp.score + blank_score
            new_hyps.append(hyp)

            topk_scores, topk_ids = (
                logp.squeeze(1).squeeze(1).topk(beam_size + 1, dim=-1)
            )

            for i in range(topk_ids.size(-1)):
                token = topk_ids[:, i]
                if (token == blank_id).all():
                    continue

                pred_out, pred_state = predictor.step(token, state=hyp.pred_state)

                new_hyps.append(
                    Hypothesis(
                        tokens=torch.cat([hyp.tokens, token.unsqueeze(-1)], dim=-1),
                        pred_out=pred_out,
                        pred_state=pred_state,
                        score=hyp.score + topk_scores[:, i],
                    )
                )
        new_hyps = sorted(
            new_hyps,
            key=lambda hyp: hyp.score.mean().item(),
            reverse=True,
        )[:beam_size]
        return new_hyps
