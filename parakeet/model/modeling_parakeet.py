import torch
from torch import nn, Tensor
from huggingface_hub import hf_hub_download
from safetensors.torch import safe_open
from parakeet.model.config import ModelConfig
from parakeet.model.encoder import ConformerEncoder
from parakeet.model.decoder import Predictor, Joiner
from parakeet.model.tokenizer import ParakeetTokenizer
from parakeet.model.feature_extractor import ParakeetFeatureExtractor
from parakeet.model.generation import GenerationMixin


def remap_weights(state_dict:dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        if 'lstm' in k:
            new_state_dict[k.replace('.lstm','')] = v
        else:
            new_state_dict[k] = v
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
        )

        self.decoder = Predictor(
            pred_dim=config.pred_hidden_dim,
            hidden_dim = config.pred_hidden_dim,
            num_layers=config.pred_hidden_layers,
            vocab_size=config.vocab_size,
        )

        self.joint = Joiner(
            encoder_dim=config.enc_hidden_dim,
            pred_dim=config.pred_hidden_dim,
            joint_dim=config.joint_hidden_dim,
            num_classes=config.vocab_size
        )

        self.blank_id = config.vocab_size

    def forward(
            self,
            audio_features: Tensor,
            labels: Tensor,
            audio_lens: Tensor,
            label_lens: Tensor | None = None,
    ):
        encoder_outputs = self.encoder(audio_features, audio_lens)
        predictor_outputs, label_lens, _ = self.decoder(labels, label_lens)
        logits = self.joint(encoder_outputs, predictor_outputs)
        return {"loss": None, "logits": logits}

    def get_feature_extractor(self):
        return self._feature_extractor

    def get_tokenizer(self):
        return self._tokenizer

    def get_joiner(self):
        return self.joint

    def get_predictor(self):
        return self.decoder

    @classmethod
    def from_pretrained(
            cls,
    ):
        config = ModelConfig()

        repo_id = 'odunola/parakeet-EOU'
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename='model.safetensors'
        )
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename='tokenizer.model'
        )


        state_dict = dict()
        with safe_open(model_path, framework='pt', device='cpu') as fp:
            for key in fp.keys():
                state_dict[key] = fp.get_tensor(key)

        model = cls(config)
        state_dict = remap_weights(state_dict)
        model.load_state_dict(state_dict)
        model._feature_extractor = ParakeetFeatureExtractor()
        model._tokenizer = ParakeetTokenizer(tokenizer_path)

        return model

if __name__ == "__main__":
    config = ModelConfig()
    model = Parakeet.from_pretrained()
    print(model)

    import librosa
    path = '/Users/odunolajenrola/Documents/GitHub/parakeet-streaming/fugitivepieces_03_pope_64kb.mp3'
    array, sr = librosa.load(path, sr=16000)
    output = model._greedy_decode(array)
    print(output)