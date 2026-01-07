from typing import Literal
import torch
from torch import nn, Tensor
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open
from parakeet.model.config import ModelConfig, LargeModelConfig
from parakeet.model.encoder import ConformerEncoder
from parakeet.model.decoder import Predictor, Joiner
from parakeet.model.tokenizer import ParakeetTokenizer
from parakeet.model.feature_extractor import ParakeetFeatureExtractor
from parakeet.model.generation import GenerationMixin

WEIGHTS_MAP = {
    "large": "odunola/Nemo-Speech-Large",
    "small": "odunola/parakeet-EOU",
}
CONFIG_MAP = {
    "large": LargeModelConfig,
    "small": ModelConfig,
}


def remap_streaming_modules(state_dict: dict):
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("decoder."):
            remapped_key = "predictor." + key[len("decoder.") :]
        elif key.startswith("joint."):
            remapped_key = "joiner." + key[len("joint.") :]
        else:
            remapped_key = key
        remapped[remapped_key] = value
    return remapped


def remap_weights(state_dict: dict):
    new_state_dict = dict()
    layer_state_dict = dict()
    for k, v in state_dict.items():
        if any(x in k for x in ["linear_k", "linear_v", "linear_q"]):
            layer_state_dict[k] = v
        elif "lstm" in k:
            new_state_dict[k.replace(".lstm", "")] = v
        else:
            new_state_dict[k] = v

    for layer_idx in range(len(layer_state_dict) // 3):
        linear_q = f"encoder.layers.{layer_idx}.self_attn.linear_q.weight"
        linear_v = f"encoder.layers.{layer_idx}.self_attn.linear_v.weight"
        linear_k = f"encoder.layers.{layer_idx}.self_attn.linear_k.weight"

        comb_weight = torch.cat(
            [
                layer_state_dict[linear_q],
                layer_state_dict[linear_k],
                layer_state_dict[linear_v],
            ]
        )
        new_state_dict[f"encoder.layers.{layer_idx}.self_attn.qkv.weight"] = comb_weight
    new_state_dict = remap_streaming_modules(new_state_dict)
    return new_state_dict


class Parakeet(nn.Module, GenerationMixin):
    def __init__(self, config: ModelConfig):
        super(Parakeet, self).__init__()
        self.config = config

        self.encoder = ConformerEncoder(
            feat_in=config.feat_in,
            hidden_size=config.enc_hidden_dim,
            ff_expansion_factor=config.ff_expansion_factor,
            num_heads=config.enc_num_heads,
            num_layers=config.enc_hidden_layers,
            subsampling_factor=config.subsampling_factor,
            pos_emb_max_len=config.pos_emb_max_len,
            conv_kernel_size=config.conv_kernel_size,
            att_context_size=config.enc_att_context_size,
            subsampling_conv_channels=config.subsampling_conv_channels,
            use_bias=config.use_bias,
            stream=config.stream,
        )

        self.predictor = Predictor(
            pred_dim=config.pred_hidden_dim,
            hidden_dim=config.pred_hidden_dim,
            num_layers=config.pred_hidden_layers,
            vocab_size=config.vocab_size,
        )

        self.joiner = Joiner(
            encoder_dim=config.enc_hidden_dim,
            pred_dim=config.pred_hidden_dim,
            joint_dim=config.joint_hidden_dim,
            num_classes=config.vocab_size,
        )

        self.blank_id = config.vocab_size

    def forward(
        self,
        audio_features: Tensor,
        labels: Tensor,
        audio_lens: Tensor,
        label_lens: Tensor | None = None,
    ):
        raise RuntimeError(
            "Offline forward is removed. Use encoder.forward_streaming with "
            "predictor.step and joiner.forward_frame."
        )

    def get_feature_extractor(self):
        return self._feature_extractor

    def get_tokenizer(self):
        return self._tokenizer

    def get_joiner(self):
        return self.joiner

    def get_predictor(self):
        return self.predictor

    @classmethod
    def from_pretrained(cls, size: Literal["small", "large"] | ModelConfig = "small"):
        if isinstance(size, ModelConfig):
            config = size
            size = "large" if isinstance(config, LargeModelConfig) else "small"
        else:
            config = CONFIG_MAP[size]()

        repo_id = WEIGHTS_MAP[size]
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.model")

        state_dict = dict()
        with safe_open(model_path, framework="pt", device="cpu") as fp:
            for key in fp.keys():
                state_dict[key] = fp.get_tensor(key)

        model = cls(config)
        state_dict = remap_weights(state_dict)
        model.load_state_dict(state_dict)
        print("loaded model")
        model._feature_extractor = ParakeetFeatureExtractor()
        model._tokenizer = ParakeetTokenizer(tokenizer_path)

        return model
