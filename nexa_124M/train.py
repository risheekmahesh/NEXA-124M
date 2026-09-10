"""Train the GPT-style model on the TinyStories dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from model import GPTModel, ModelConfig


class TokenWindowDataset(Dataset):
    """Turn one token stream into input and next-token target windows."""

    def __init__(self, token_ids: list[int], context_length: int, stride: int):
        self.inputs = []
        self.targets = []
        for start in range(0, len(token_ids) - context_length - 1, stride):
            window = token_ids[start : start + context_length + 1]
            self.inputs.append(torch.tensor(window[:-1], dtype=torch.long))
            self.targets.append(torch.tensor(window[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text(dataset_name: str, split: str, max_stories: int | None) -> str:
    dataset = load_dataset(dataset_name, split=split)
    if max_stories is not None:
        dataset = dataset.select(range(min(max_stories, len(dataset))))
    return "\n\n".join(dataset["text"])


def encode_text(text: str, tokenizer) -> list[int]:
    return tokenizer.encode(text, allowed_special={"<|endoftext|>"})


def make_loaders(train_tokens: list[int], valid_tokens: list[int], config: ModelConfig, batch_size: int, stride: int):
    train_set = TokenWindowDataset(train_tokens, config.context_length, stride)
    valid_set = TokenWindowDataset(valid_tokens, config.context_length, stride)
    if not train_set or not valid_set:
        raise ValueError("Not enough tokens for the selected context length")
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True),
    )


def loss_on_batch(model: GPTModel, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(inputs)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())


def evaluate(model: GPTModel, loader: DataLoader, device: torch.device, batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for index, (inputs, targets) in enumerate(loader):
            if index >= batches:
                break
            losses.append(loss_on_batch(model, inputs.to(device), targets.to(device)).item())
    model.train()
    return float(np.mean(losses))


def train(model: GPTModel, train_loader: DataLoader, valid_loader: DataLoader, *, device: torch.device,
          epochs: int, learning_rate: float, eval_batches: int, log_every: int) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    history = []
    step = 0
    model.to(device)
    for epoch in range(1, epochs + 1):
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on_batch(model, inputs.to(device), targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % log_every == 0:
                validation_loss = evaluate(model, valid_loader, device, eval_batches)
                record = {"step": step, "train_loss": loss.item(), "validation_loss": validation_loss}
                history.append(record)
                print(f"step {step:>5} | train loss {loss.item():.4f} | validation loss {validation_loss:.4f}")
    return history


def save_checkpoint(path: Path, model: GPTModel, history: list[dict], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config_dict(), "history": history}, path)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    path.with_suffix(".json").write_text(json.dumps(arguments, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NEXA on TinyStories")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--max-stories", type=int, default=2_000, help="Small default keeps the first run manageable")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tinystories.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer = tiktoken.get_encoding("gpt2")
    config = ModelConfig(vocab_size=tokenizer.n_vocab, context_length=args.context_length)
    print(f"Loading {args.dataset}...")
    train_text = load_text(args.dataset, args.train_split, args.max_stories)
    valid_text = load_text(args.dataset, args.validation_split, max(100, args.max_stories // 10))
    train_tokens = encode_text(train_text, tokenizer)
    valid_tokens = encode_text(valid_text, tokenizer)
    train_loader, valid_loader = make_loaders(train_tokens, valid_tokens, config, args.batch_size, args.stride)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(config)
    print(f"device: {device} | parameters: {model.parameter_count():,}")
    history = train(model, train_loader, valid_loader, device=device, epochs=args.epochs,
                    learning_rate=args.learning_rate, eval_batches=args.eval_batches, log_every=args.log_every)
    save_checkpoint(args.checkpoint, model, history, args)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
