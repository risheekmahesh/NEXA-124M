"""Model definition for a small GPT-style language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 50_257
    context_length: int = 256
    emb_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    drop_rate: float = 0.1
    qkv_bias: bool = False


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.emb_dim
        self.layers = nn.Sequential(
            nn.Linear(config.emb_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, config.emb_dim),
            nn.Dropout(config.drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.emb_dim % config.n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")

        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.qkv = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.qkv_bias)
        self.out = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)
        mask = torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, emb_dim = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        shape = (batch_size, sequence_length, self.n_heads, self.head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        scores = scores.masked_fill(self.mask[:sequence_length, :sequence_length], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        output = (weights @ value).transpose(1, 2).contiguous().view(batch_size, sequence_length, emb_dim)
        return self.out(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.emb_dim)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.emb_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.emb_dim)
        self.embedding_dropout = nn.Dropout(config.drop_rate)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layers)))
        self.final_norm = nn.LayerNorm(config.emb_dim)
        self.output_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError("Input is longer than the configured context length")
        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        x = self.blocks(x)
        return self.output_head(self.final_norm(x))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.config)
