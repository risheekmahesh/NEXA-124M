# NEXA-124M

NEXA stands for **Neural Exchange and Reasoning Architecture**. This is my attempt to understand how a small GPT-style language model works by implementing the main pieces myself in PyTorch.

This repository started as a messy Colab notebook. I was following along with the basics of language models, testing tokenization, attention, Transformer blocks, and text generation. The first version did not run from start to finish, so I cleaned it up into a small project that I can actually explain and run.

This is **not a finished chatbot** and it is not a pretrained model. The useful part of the project is the implementation and the learning process.

## What works right now

The current version can:

- tokenize text with the GPT-2 tokenizer from `tiktoken`;
- create next-token prediction batches;
- run tokens through a decoder-only Transformer;
- calculate a cross-entropy loss;
- generate tokens using greedy decoding or sampling; and
- run a small CPU demonstration.

The default demo uses a small model because I wanted something that could run locally without a GPU. There is also a `gpt2-small` configuration with roughly 124 million parameters, but defining that model is very different from successfully pretraining it.

## A quick look at the model

The model is made up of the same basic pieces I was trying to understand in the original notebook:

1. token embeddings and positional embeddings;
2. masked multi-head self-attention;
3. feed-forward layers with GELU;
4. residual connections and layer normalization; and
5. a final projection to vocabulary logits.

More detail is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). I kept the code split into a few files so it is easier to read than the original notebook.

## Run it

```bash
git clone https://github.com/risheekmahesh/NEXA-124M-clean.git
cd NEXA-124M-clean
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/quickstart.py
```

To run the tests:

```bash
python -m pytest -q
```

The generated text from the quickstart will not be sensible because the model starts with random weights. That is expected. The quickstart is mainly checking that the forward pass, loss, and generation path work.

## Model sizes

| Name | Layers | Hidden size | Attention heads | Why it is here |
| --- | ---: | ---: | ---: | --- |
| `tiny` | 2 | 128 | 4 | Runs quickly on a CPU for demonstrations |
| `gpt2-small` | 12 | 768 | 12 | Approximate 124M-parameter architecture |

Example:

```python
from nexa import GPTModel
from configs.model_configs import get_config

model = GPTModel(get_config("tiny"))
print(model.num_parameters())
```

## What I learned from the first attempt

The original notebook had several problems that made it hard to present or reproduce:

- package installation commands were mixed into code cells;
- some cells depended on variables created much earlier;
- the training text was downloaded in the middle of the notebook;
- different configuration values were used at different points; and
- a model being created successfully was confused with a model being trained successfully.

This repository is my second pass. It is deliberately smaller. I would rather have a version with a working smoke test and clearly stated limitations than claim that I trained a useful 124M model when I did not.

## Limitations and next steps

There is no pretrained checkpoint in this repository. I have not included a complete long-running pretraining script, distributed training, instruction tuning, or a chat interface. The next useful step would be to add a small training script and record loss curves on a clearly documented dataset.

I also want to compare the attention implementation against a few hand-calculated examples instead of only checking tensor shapes. That would make the project a better learning exercise.

## References

The model design is based on the Transformer architecture and GPT-style autoregressive language modeling:

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
[2]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf "Language Models are Unsupervised Multitask Learners"

## License

MIT. See [`LICENSE`](LICENSE).
