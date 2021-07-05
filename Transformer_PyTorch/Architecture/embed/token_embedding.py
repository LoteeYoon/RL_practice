import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    # dense representation of word using weighted matrix

    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)
        # vocab_size : number of words to embed
        # d_model : dim of embedding vector
        # padding_idx : index of token to pad
        # nn.Embedding : return embedding vector [vocab_size, d_model]
