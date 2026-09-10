from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    """Create overlapping next-token prediction windows from a text string."""

    def __init__(self, text: str, tokenizer, max_length: int, stride: int) -> None:
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        self.input_ids = []
        self.target_ids = []
        for start in range(0, len(token_ids) - max_length, stride):
            chunk = token_ids[start : start + max_length + 1]
            self.input_ids.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.target_ids.append(torch.tensor(chunk[1:], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(text: str, tokenizer, *, batch_size: int = 2, max_length: int = 128,
                      stride: int = 128, shuffle: bool = True) -> DataLoader:
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
