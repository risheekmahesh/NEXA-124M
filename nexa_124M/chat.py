"""Generate text from a trained NEXA checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import tiktoken

from model import GPTModel, ModelConfig


def generate(model: GPTModel, input_ids: torch.Tensor, max_new_tokens: int, *, temperature: float, top_k: int | None) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -model.config.context_length :]
            logits = model(context)[:, -1, :]
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float("-inf")
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained NEXA model")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tinystories.pt"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["config"])
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model"])
    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    output = generate(model, input_ids, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
