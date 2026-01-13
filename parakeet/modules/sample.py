from abc import ABC

import torch
from torch import Tensor


class GenerationMixin(ABC):
    max_symbols_per_timestep: int = 10

    def _greedy_decode_frame(
        self,
        frame: Tensor,
        pred_out: Tensor,
        pred_state: Tensor,
    ):
        predictor = self.predictor
        joiner = self.joiner
        max_symbols = self.max_symbols_per_timestep
        blank_id = self.blank_id

        batch_size = frame.size(0)
        tokens = torch.empty(
            (batch_size, max_symbols), dtype=torch.long, device=frame.device
        )
        out_len = 0
        for _ in range(max_symbols):
            logits = joiner.forward_frame(frame, pred_out)
            ids = logits.argmax(-1)
            if (ids == blank_id).all():
                break
            tokens[:, out_len] = ids
            out_len += 1
            pred_out, pred_state = predictor.step(ids, state=pred_state)

        return tokens[:, :out_len], pred_out, pred_state
