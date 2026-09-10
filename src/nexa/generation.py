from __future__ import annotations

import torch


def generate(model, input_ids: torch.Tensor, max_new_tokens: int, *, context_length: int,
             temperature: float = 0.0, top_k: int | None = None, eos_id: int | None = None) -> torch.Tensor:
    """Generate tokens with optional temperature and top-k sampling."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -context_length:]
            logits = model(context)[:, -1, :]
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float("-inf")
                next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if eos_id is not None and torch.all(next_id == eos_id):
                break
    return input_ids


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
