from dataclasses import dataclass
from parakeet.model.config import ModelConfig


@dataclass
class Config:
    model_config: ModelConfig
    num_concurrent_requests: int = 50
