import numpy as np
from parakeet.model.encoder import ConformerEncoder
from parakeet.model.naive_cache import ModelCache
from parakeet.model.feature_extractor import ParakeetFeatureExtractor


feature_extractor = ParakeetFeatureExtractor()
encoder = ConformerEncoder(
    feat_in=128,
    hidden_size=512,
    ff_expansion_factor=4,
    num_layers=17,
    num_heads=8,
    subsampling_factor=8,
    pos_emb_max_len=5000,
    conv_kernel_size=9,
    att_context_size=[70, 1],
    subsampling_conv_channels=256,
    use_bias=False,
).eval()
model_cache = ModelCache()

CHUNK_SIZE = 2548
ARRAY_SIZE = 203840
print(ARRAY_SIZE)
array = np.random.randn(ARRAY_SIZE)
features = []

total_num_frames = 0
for start in range(0, ARRAY_SIZE, CHUNK_SIZE):
    chunk = array[start : start + CHUNK_SIZE]
    assert chunk.shape == (CHUNK_SIZE,)
    feat = feature_extractor(chunk)["input_features"]
    frames = feat.shape[-1]
    total_num_frames += frames
    features.append(feat)

print("naive_run", feature_extractor(array)["input_features"].shape)
print("Got seq len ", total_num_frames)
