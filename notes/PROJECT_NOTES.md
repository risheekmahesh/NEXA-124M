# Project notes

I started this project because I wanted to understand what is happening inside a language model instead of only calling an existing API.

The first version was a Google Colab notebook. I copied ideas from different parts of the learning process into one place: splitting text into tokens, adding special tokens, trying `tiktoken`, building a sliding-window dataset, implementing attention, defining Transformer blocks, and eventually attempting text generation.

That notebook was useful for experimenting, but it was not a good final project. It had installation commands inside cells, repeated configuration dictionaries, variables that only existed if earlier cells had been run in the correct order, and a training section that I never completed properly.

For this version I made a smaller target. I wanted to make sure that a model could be instantiated, accept token IDs, return logits with the expected shape, calculate a loss, and generate a sequence. Those checks now run in the test suite and in the quickstart example.

The random output from the tiny model is not a success metric. It is only evidence that the code path runs. To claim that the model has learned language, I still need to train it on a documented text corpus and show training and validation curves.

## Questions I still want to answer

- How quickly does the loss decrease on a very small corpus?
- How much does the context length affect the result?
- Can I verify the causal mask with a simple hand-designed example?
- How much memory does the 124M configuration use during training?
- What changes when the model is trained from scratch versus loaded from an existing checkpoint?
