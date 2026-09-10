from __future__ import annotations

import torch
from torch import nn


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh((2.0 / torch.pi) ** 0.5 * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hidden = 4 * cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], hidden), GELU(), nn.Linear(hidden, cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        assert cfg["emb_dim"] % cfg["n_heads"] == 0
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.qkv = nn.Linear(cfg["emb_dim"], 3 * cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.register_buffer("mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, tokens, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        scores = scores.masked_fill(self.mask[:tokens, :tokens], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.attn = MultiHeadCausalAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class GPTModel(nn.Module):
    """Decoder-only Transformer for autoregressive next-token prediction."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg.copy()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(*(TransformerBlock(cfg) for _ in range(cfg["n_layers"])))
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError(f"Sequence length {seq_len} exceeds context length {self.cfg['context_length']}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.drop_emb(self.tok_emb(input_ids) + self.pos_emb(positions))
        return self.out_head(self.final_norm(self.blocks(x)))

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
