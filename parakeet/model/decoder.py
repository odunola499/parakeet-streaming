import torch
from torch import nn, Tensor


class Predictor(nn.Module):
    def __init__(
        self, pred_dim: int, hidden_dim: int, num_layers: int, vocab_size: int
    ):
        super(Predictor, self).__init__()
        self.prediction = nn.ModuleDict(
            {
                "embed": nn.Embedding(vocab_size + 1, pred_dim, padding_idx=vocab_size),
                "dec_rnn": nn.LSTM(
                    input_size=pred_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                ),
            }
        )
        self.pred_dim = pred_dim
        self.num_layers = num_layers

    def init_state(self, batch_size: int):
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype
        return (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.pred_dim,
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                self.num_layers,
                batch_size,
                self.pred_dim,
                device=device,
                dtype=dtype,
            ),
        )

    def forward(self, targets: Tensor, target_length: Tensor, states=None):
        y = self.prediction["embed"](targets)

        bsz, seq_len, hidden = y.shape
        start = torch.zeros(
            (bsz, 1, hidden), dtype=targets.dtype, device=targets.device
        )
        y = torch.concat([start, y], dim=1).contiguous()

        if states is None:
            states = self.init_state(bsz)

        y = y.transpose(0, 1)
        g, hidden = self.prediction["dec_rnn"](y, states)
        g = g.transpose(0, 1).transpose(1, 2)
        return g, target_length, hidden

    def step(self, input_ids: Tensor, state: tuple[Tensor, Tensor] | Tensor):
        input_ids = input_ids.reshape(-1, 1)
        y = self.prediction["embed"](input_ids)
        y = y.transpose(0, 1)
        g, hidden = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)
        return g[:, -1, :], hidden


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim,
        pred_dim,
        joint_dim,
        dropout: float = 0.2,
        num_classes: int = 1024,
    ):
        super().__init__()

        self._vocab_size = num_classes
        self._num_classes = num_classes + 1
        self.encoder_dim = encoder_dim
        self.pred_dim = pred_dim
        self.joint_dim = joint_dim
        self.dropout = dropout

        self.pred = nn.Linear(pred_dim, joint_dim)
        self.enc = nn.Linear(encoder_dim, joint_dim)
        activation = nn.ReLU(inplace=True)

        layers = (
            [activation]
            + [nn.Dropout(p=dropout)]
            + [nn.Linear(joint_dim, num_classes + 1)]
        )
        self.joint_net = nn.Sequential(*layers)

    def forward(
        self,
        encoder_output: Tensor,  # B, d1, T
        predictor_output: Tensor,  # B, d2, U
    ):

        if encoder_output.shape[-1] == self.encoder_dim:
            pass
        elif encoder_output.shape[1] == self.encoder_dim:
            encoder_output = encoder_output.transpose(1, 2)
        else:
            raise ValueError("encoder_output last dim must match encoder_dim")

        if predictor_output.dim() == 2:
            predictor_output = predictor_output.unsqueeze(1)
        elif predictor_output.dim() != 3:
            raise ValueError(
                "predictor_output must have shape (B, D), (B, U, D), or (B, D, U)"
            )

        if predictor_output.shape[-1] == self.pred_dim:
            pass
        elif predictor_output.shape[1] == self.pred_dim:
            predictor_output = predictor_output.transpose(1, 2)
        else:
            raise ValueError("predictor_output last dim must match pred_dim")

        encoder_output = self.enc(encoder_output).unsqueeze(2)
        decoder_output = self.pred(predictor_output).unsqueeze(1)
        joined = encoder_output + decoder_output
        return self.joint_net(joined)

    def forward_frame(self, encoder_frame: Tensor, predictor_out: Tensor) -> Tensor:
        encoder_proj = self.enc(encoder_frame)
        predictor_proj = self.pred(predictor_out).unsqueeze(1)
        joined = encoder_proj + predictor_proj
        return self.joint_net(joined).squeeze(1)
