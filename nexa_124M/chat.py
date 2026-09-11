"""Inference and decoding code extracted from the LLM_from_scratch notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import tiktoken

from model import GPTModel


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy generation from notebook section 4.7."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.8, top_k=40,
             eos_id=None, repetition_penalty=1.1, min_new_tokens=20):
    """Notebook section 5.3: temperature and top-k decoding."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        if repetition_penalty != 1.0:
            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty
        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_top_logit = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_logit, torch.tensor(float("-inf"), device=logits.device), logits)
        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        generated = idx.shape[1] - 1
        if eos_id is not None and generated >= min_new_tokens and torch.all(idx_next == eos_id):
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = GPTModel(checkpoint["config"])
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    model.load_state_dict(state_dict)
    return model, checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained notebook GPT model")
    parser.add_argument("--checkpoint", type=Path, default=Path("../checkpoints/tinystories.pt"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--min-tokens", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = tiktoken.get_encoding("gpt2")
    model, _ = load_checkpoint(args.checkpoint)
    token_ids = text_to_token_ids(args.prompt, tokenizer)
    output = generate(
        model, token_ids, args.tokens, model.cfg["context_length"],
        args.temperature, args.top_k, eos_id=tokenizer.eot_token,
        repetition_penalty=args.repetition_penalty, min_new_tokens=args.min_tokens,
    )
    print(token_ids_to_text(output, tokenizer))


if __name__ == "__main__":
    main()
