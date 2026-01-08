from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    feat_in: int = 128
    enc_hidden_dim: int = 512
    enc_hidden_layers: int = 17
    enc_num_heads: int = 8
    enc_att_context_size: tuple[int, int] = (70, 1)
    pos_emb_max_len: int = 5000
    ff_expansion_factor: int = 4
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    use_bias: bool = False

    pred_hidden_dim: int = 640
    pred_hidden_layers: int = 1
    joint_hidden_dim: int = 640
    vocab_size: int = 1026

    stream: bool = True


@dataclass
class LargeModelConfig(ModelConfig):
    enc_hidden_dim: int = 1024
    enc_hidden_layers: int = 24
    enc_att_context_size: tuple[int, int] = (70, 13)
    pred_hidden_layers: int = 2
    vocab_size: int = 1024


@dataclass
class Config:
    size: Literal["small", "large"] = "small"
    num_concurrent_requests: int = 50
    max_num_streams: int = 100
    sample_rate: int = 16000
    max_stream_seconds: int = 300
    max_seq_len: int = 5000

    def __post_init__(self):
        if self.size == "small":
            self.model_config = ModelConfig()
        elif self.size == "large":
            self.model_config = LargeModelConfig()
        else:
            raise NotImplementedError
