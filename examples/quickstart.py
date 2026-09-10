from pathlib import Path
import sys

import torch
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexa import GPTModel, cross_entropy_loss, generate
from configs.model_configs import get_config


def main() -> None:
    torch.manual_seed(123)
    cfg = get_config("tiny")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(cfg)
    prompt = "NEXA learns"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    logits = model(input_ids)
    targets = input_ids.roll(-1, dims=1)
    loss = cross_entropy_loss(logits, targets)
    output = generate(model, input_ids, max_new_tokens=12, context_length=cfg["context_length"], temperature=0.8, top_k=20)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Demo next-token loss: {loss.item():.4f}")
    print(f"Generated text: {tokenizer.decode(output[0].tolist())!r}")


if __name__ == "__main__":
    main()
