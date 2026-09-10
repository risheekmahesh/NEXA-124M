# NEXA-124M

**NEXA** — Neural Exchange and Reasoning Architecture — is an educational decoder-only Transformer language model implemented from first principles in PyTorch.

This repository is a cleaned and reproducible continuation of my first attempt to build a language model from scratch. It focuses on the core mechanics behind GPT-style models: token embeddings, positional embeddings, causal self-attention, feed-forward networks, residual connections, layer normalization, next-token prediction, training, and text generation.

> **Project status:** The implementation is complete as an educational GPT-style model. The default demo uses a tiny configuration so it can run on a CPU. The included `gpt2-small` configuration is approximately 124 million parameters and requires a suitable GPU, dataset, and training time for meaningful results.

## What this project demonstrates

| Area | Implementation |
| --- | --- |
| Tokenization | GPT-2 byte-pair encoding through `tiktoken` |
| Input pipeline | Sliding windows of input and target token sequences |
| Architecture | Decoder-only Transformer |
| Attention | Multi-head causal self-attention |
| Non-linearity | GELU activation |
| Optimization | AdamW with cross-entropy loss |
| Generation | Greedy or temperature/top-k sampling |
| Evaluation | Training and validation loss, plus perplexity |
| Reproducibility | Fixed seeds, configuration-driven model construction, and smoke tests |

## Repository layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── model_configs.py
├── docs/
│   └── ARCHITECTURE.md
├── examples/
│   └── quickstart.py
├── src/
│   └── nexa/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       └── generation.py
└── tests/
    └── test_model.py
```

## Quick start

The project requires Python 3.10 or newer. A virtual environment is recommended.

```bash
git clone https://github.com/risheekmahesh/NEXA-124M-clean.git
cd NEXA-124M-clean
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the CPU-friendly demonstration:

```bash
python examples/quickstart.py
```

Run the test suite:

```bash
python -m pytest
```

The quickstart constructs a small model, performs a forward pass, calculates the next-token loss, and generates a short continuation. This validates the complete model path without pretending that a tiny untrained model is a useful chatbot.

## Model configurations

The configuration module includes the following presets:

| Preset | Layers | Embedding size | Heads | Approximate parameters | Intended use |
| --- | ---: | ---: | ---: | ---: | --- |
| `tiny` | 2 | 128 | 4 | A few million | CPU demonstrations and tests |
| `gpt2-small` | 12 | 768 | 12 | 124M | Research and training experiments |

Create a model in Python:

```python
from nexa import GPTModel, get_config

model = GPTModel(get_config("tiny"))
print(f"Parameters: {model.num_parameters():,}")
```

## Training on your own text

The model learns next-token prediction from plain text. A training-ready dataset should be large enough to contain diverse examples and should be legally usable for the intended purpose.

The basic training loop is:

1. Read and clean a text corpus.
2. Split the corpus into training and validation portions.
3. Convert text into GPT-2 token IDs.
4. Build fixed-length input/target windows.
5. Feed input tokens through the Transformer.
6. Compare logits with the target tokens using cross-entropy.
7. Update weights with AdamW.
8. Monitor validation loss and generate samples periodically.

The source notebook that inspired this repository mixed implementation cells, package installation commands, downloaded data, and large-scale experiments in one file. This version separates reusable code from demonstrations so each component can be tested and explained independently.

## Limitations

NEXA-124M is an educational implementation, not a production assistant. It does not include instruction tuning, preference optimization, distributed training, tokenizer training, checkpoint conversion, safety filtering, a web interface, or a pretrained checkpoint. A randomly initialized model will generate poor text until it has been trained on a sufficiently large corpus.

The 124M configuration is intentionally provided as an architecture target. Running it end-to-end is compute-intensive, and a successful forward pass is not equivalent to successful pretraining.

## Learning references

The implementation follows the standard decoder-only Transformer design introduced by Vaswani et al. and the GPT-2-style language-modeling approach. The repository is intended to be read alongside the source code and architecture notes in `docs/ARCHITECTURE.md`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
[2]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf "Language Models are Unsupervised Multitask Learners"
