# NEXA-124M

This is my small GPT-style language model project trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

The original learning notebook is included in [`notebooks/llmfromscratch.ipynb`](notebooks/llmfromscratch.ipynb). It contains the longer experiments that led to this project: manual tokenization, vocabulary construction, sliding-window sampling, several attention versions, Transformer blocks, loss calculations, and early training attempts. The notebook is kept as a record of the work, while the Python files are the clean runnable version.

I reorganized the code so the main workflow is easy to follow:

- `model.py` contains the neural-network classes only.
- `train.py` contains the TinyStories data loading, batching, loss, optimizer, training loop, evaluation, and checkpoint saving.
- `chat.py` contains checkpoint loading and text generation only.

The default model is intentionally smaller than 124M parameters so that I can test the complete workflow. The model is still GPT-like, but a model definition is not the same thing as a successfully pretrained 124M model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train on TinyStories

The dataset is downloaded automatically through the Hugging Face `datasets` library.

For a small first run:

```bash
cd nexa_124M
python train.py --max-stories 200 --context-length 64 --batch-size 4 --epochs 1 --log-every 10
```

The checkpoint is saved to:

```text
checkpoints/tinystories.pt
```

For a larger run, increase `--max-stories`, `--context-length`, and `--epochs`. The full TinyStories dataset is large, so training it from scratch can take a long time and needs suitable hardware.

## Generate text

After training:

```bash
cd nexa_124M
python chat.py --checkpoint checkpoints/tinystories.pt --prompt "Once upon a time" --tokens 100
```

Temperature can be set to `0` for greedy generation or to a value such as `0.8` for sampling:

```bash
python chat.py --checkpoint checkpoints/tinystories.pt --prompt "The little dragon" --temperature 0.8 --top-k 40
```

## File organization

`model.py` defines the model configuration, GELU, feed-forward network, causal multi-head self-attention, Transformer block, and GPT model. It does not download data or run training.

`train.py` downloads the `roneneldan/TinyStories` dataset, tokenizes the stories with the GPT-2 tokenizer, creates input/target windows, trains with AdamW, prints training and validation loss, and saves a checkpoint.

`chat.py` loads the saved model configuration and weights, encodes a prompt, and generates new tokens. It does not import the dataset or optimizer.

## Notes

The model starts with random weights. A very short training run is mainly a test that the pipeline works. The generated stories will improve only after training for long enough on enough data.

## Notebook versus Python files

The notebook is exploratory and follows the order in which I learned the ideas. It includes intermediate outputs and some cells that depend on earlier cells. It is useful for showing the learning process, but it is not the most reliable way to run the model.

The Python version removes that hidden notebook state. `train.py` contains the reusable versions of dataset creation, loss calculation, evaluation, training, and checkpoint saving. `chat.py` contains only inference. This is the version to use from a terminal.

The GPT-2 tokenizer has a vocabulary of 50,257 tokens. The default model uses four Transformer blocks, a 256-dimensional embedding, four attention heads, and a context length of 256. These values can be changed in `model.py` or exposed as command-line options in `train.py`.

## Reference

The architecture follows the decoder-only Transformer design from *Attention Is All You Need* [1] and the GPT-style autoregressive language-modeling approach [2].

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
[2]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf "Language Models are Unsupervised Multitask Learners"
