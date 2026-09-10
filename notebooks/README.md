# Notebook history

`LLM_from_scratch.ipynb` is the main learning notebook for this project. It contains the original experiments from text preprocessing through attention, Transformer blocks, GPT model construction, loss calculation, decoding, and pretraining.

I also added a final TinyStories section. That section uses the clean code in `../nexa_124M/` to download TinyStories, create token windows, train the model, and save a checkpoint for `chat.py`.

The notebook is intentionally exploratory. For a reliable terminal workflow, use:

- `../nexa_124M/model.py` — model classes
- `../nexa_124M/train.py` — data preparation and training
- `../nexa_124M/chat.py` — inference
