import sys
from pathlib import Path

import torch
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.model_configs import get_config
from nexa import GPTModel, cross_entropy_loss, generate


def test_forward_shape_and_loss():
    cfg = get_config("tiny")
    model = GPTModel(cfg)
    tokens = torch.randint(0, cfg["vocab_size"], (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, cfg["vocab_size"])
    assert torch.isfinite(cross_entropy_loss(logits, tokens))


def test_generation_extends_sequence():
    cfg = get_config("tiny")
    model = GPTModel(cfg)
    prompt = torch.tensor([[1, 2, 3]])
    result = generate(model, prompt, 4, context_length=cfg["context_length"])
    assert result.shape == (1, 7)


def test_tokenizer_round_trip_is_available():
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "NEXA from scratch"
    assert tokenizer.decode(tokenizer.encode(text)) == text
