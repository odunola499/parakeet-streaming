from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    feat_in: int = 128
    enc_hidden_dim: int = 512
    pred_hidden_dim: int = 640
    joint_hidden_dim: int = 640

    enc_hidden_layers: int = 17
    enc_num_heads: int = 8
    enc_att_context_size: list = field(default_factory=lambda: [70, 1])
    pos_emb_max_len: int = 5000
    pred_hidden_layers: int = 1
    use_bias: bool = False
    ff_expansion_factor: int = 4
    conv_kernel_size: int = 9

    enc_use_bias: bool = False
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256

    dropout: float = 0.2
    dropout_pre_encoder: float = 0.1
    dropout_emb: float = 0.0
    dropout_att: float = 0.1
    dec_dropout: float = 0.2
    vocab_size: int = 1026

    # Streaming config
    stream: bool = False
    chunk_size: list = field(default_factory=lambda: [9, 16])
    shift_size: list = field(default_factory=lambda: [9, 16])
    cache_drop_size: int = 0
    last_channel_cache_size: int = 70
    valid_out_len: int = 2
    pre_encode_cache_size: list = field(default_factory=lambda: [0, 9])
    drop_extra_pre_encoded: int = 2
    last_channel_num: int = 0
    last_time_num: int = 0
