# Architecture notes

NEXA is a decoder-only Transformer. Each input token is mapped to a learned token embedding and added to a learned positional embedding. The resulting sequence passes through repeated Transformer blocks.

Each block applies pre-normalized causal self-attention followed by a feed-forward network. Residual connections preserve the input to each sublayer. Causal masking prevents a token from attending to future positions, which makes the model suitable for autoregressive generation.

The final normalized hidden state is projected to vocabulary-sized logits. During training, the logits at position *t* are compared with the token at position *t + 1*. Cross-entropy therefore trains the model to estimate the next-token distribution.

| Component | Role |
| --- | --- |
| Token embedding | Converts token IDs into vectors. |
| Positional embedding | Encodes token order. |
| Causal attention | Mixes information from the current and previous tokens only. |
| Feed-forward network | Applies a position-wise nonlinear transformation. |
| Layer normalization | Stabilizes activations during optimization. |
| Output projection | Converts hidden states into vocabulary logits. |

The implementation intentionally uses explicit PyTorch modules rather than hiding the architecture behind a high-level language-model API. This makes tensor shapes and data flow inspectable for learning and presentation.
