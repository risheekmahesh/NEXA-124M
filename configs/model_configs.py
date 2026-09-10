from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50_257
    context_length: int = 256
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


PRESETS = {
    "tiny": ModelConfig(vocab_size=50_257, context_length=128, emb_dim=128, n_heads=4, n_layers=2, drop_rate=0.0),
    "gpt2-small": ModelConfig(vocab_size=50_257, context_length=1_024, emb_dim=768, n_heads=12, n_layers=12, drop_rate=0.1, qkv_bias=True),
}


def get_config(name: str = "tiny") -> dict:
    """Return a model configuration as a plain dictionary."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name].as_dict()
