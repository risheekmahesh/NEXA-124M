# Notebook history

`LLM_from_scratch.ipynb` is the main learning notebook for this project. It contains the original experiments from text preprocessing through attention, Transformer blocks, GPT model construction, loss calculation, decoding, and pretraining.

The files in `../nexa_124M/` are a direct separation of the notebook code so the model can be run without executing cells in order:

- `model.py` — the notebook model classes and configurations
- `train.py` — the notebook dataset, loss, evaluation, and training functions
- `chat.py` — the notebook generation and decoding functions

Only notebook-specific setup was changed, such as turning variables into command-line arguments and adding checkpoint loading.
