"""Training code extracted from the LLM_from_scratch notebook."""

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

from model import GPTModel, GPT_CONFIG_124M, TINY_CONFIG


class GPTDatasetV1(Dataset):
    """Sliding-window input/target dataset used in the notebook."""

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, tokenizer, batch_size=2, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_text(dataset_name="roneneldan/TinyStories", split="train", max_stories=None):
    dataset = load_dataset(dataset_name, split=split)
    if max_stories is not None:
        dataset = dataset.select(range(min(max_stories, len(dataset))))
    return "\n\n".join(dataset["text"])


def encode_text(text, tokenizer):
    return tokenizer.encode(text, allowed_special={"<|endoftext|>"})


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total_loss = 0.0
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    token_ids = generate_text_simple(model, encoded, max_new_tokens=50, context_size=model.cfg["context_length"])
    print(token_ids_to_text(token_ids.cpu(), tokenizer).replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def save_checkpoint(path, model, train_losses, val_losses, tokens_seen):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the notebook GPT model on TinyStories")
    parser.add_argument("--model-size", choices=["tiny", "124m"], default="tiny")
    parser.add_argument("--max-stories", type=int, default=200)
    parser.add_argument("--validation-stories", type=int, default=100)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-iter", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", type=Path, default=Path("../checkpoints/tinystories.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    tokenizer = tiktoken.get_encoding("gpt2")
    config = dict(TINY_CONFIG if args.model_size == "tiny" else GPT_CONFIG_124M)
    config["vocab_size"] = tokenizer.n_vocab
    config["context_length"] = args.context_length
    stride = args.stride or args.context_length
    print("Loading TinyStories...")
    train_text = load_text(split="train", max_stories=args.max_stories)
    val_text = load_text(split="validation", max_stories=args.validation_stories)
    train_loader = create_dataloader_v1(train_text, tokenizer, args.batch_size, config["context_length"], stride)
    val_loader = create_dataloader_v1(val_text, tokenizer, args.batch_size, config["context_length"], stride, shuffle=False, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    print(f"device: {device} | parameters: {model.parameter_count():,}")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, args.epochs,
        args.eval_freq, args.eval_iter, "Once upon a time", tokenizer,
    )
    save_checkpoint(args.checkpoint, model, train_losses, val_losses, tokens_seen)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
