import torch.nn as nn
from Architecture.embed.token_embedding import TokenEmbedding
from Architecture.embed.positional_encoding import PositionalEncoding


class Embedding(nn.Module):
    # token embedding + positional encoding
    def __init__(self, vocab_size, d_model, max_len, drop_rate, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_rate)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_enc = self.pos_enc(x)
        return self.drop_out(tok_emb + pos_enc)
