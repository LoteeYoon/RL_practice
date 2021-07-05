import torch.nn as nn
from Architecture.blocks.encoder_layer import EncoderLayer
from Architecture.embed.embedding import Embedding


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, d_ff, n_heads, n_layers, drop_rate, device):
        super().__init__()
        self.emb = Embedding(d_model=d_model,
                             max_len=max_len,
                             vocab_size=enc_voc_size,
                             drop_rate=drop_rate,
                             device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  d_ff=d_ff,
                                                  n_head=n_heads,
                                                  drop_rate=drop_rate) for _ in range(n_layers)])

    def forward(self, x, s_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, s_mask)

        return x
