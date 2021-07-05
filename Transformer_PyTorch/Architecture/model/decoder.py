import torch
import torch.nn as nn
from Architecture.blocks.decoder_layer import DecoderLayer
from Architecture.embed.embedding import Embedding


class Decoder(nn.Module):

    def __init__(self, dec_voc_size, max_len, d_model, d_ff, n_heads, n_layers, drop_rate, device):
        super().__init__()
        self.emb = Embedding(d_model=d_model,
                             drop_rate=drop_rate,
                             max_len=max_len,
                             vocab_size=dec_voc_size,
                             device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  d_ff=d_ff,
                                                  n_head=n_heads,
                                                  drop_rate=drop_rate) for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, t_mask, s_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, t_mask, s_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
