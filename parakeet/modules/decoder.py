import torch
from torch import Tensor, nn


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

    def forward(self, input_ids: Tensor, state: tuple[Tensor, Tensor] | Tensor):
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

    def forward(self, encoder_frame: Tensor, predictor_out: Tensor) -> Tensor:
        encoder_proj = self.enc(encoder_frame)
        predictor_proj = self.pred(predictor_out).unsqueeze(1)
        joined = encoder_proj + predictor_proj
        return self.joint_net(joined).squeeze(1)
