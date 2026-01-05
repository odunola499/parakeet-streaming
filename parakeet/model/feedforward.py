from torch import nn


class ConformerFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_ff: int,
        dropout: float,
        use_bias: bool = True,
    ):
        super().__init__()
        self.d_model = hidden_size
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(hidden_size, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
