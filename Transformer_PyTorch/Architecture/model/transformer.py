import torch
import torch.nn as nn
from Architecture.model.encoder import Encoder
from Architecture.model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model,
                 n_heads, max_len, d_ff, n_layers, drop_rate, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(d_model=d_model,
                               n_heads=n_heads,
                               max_len=max_len,
                               d_ff=d_ff,
                               enc_voc_size=enc_voc_size,
                               drop_rate=drop_rate,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_heads=n_heads,
                               max_len=max_len,
                               d_ff=d_ff,
                               dec_voc_size=dec_voc_size,
                               drop_rate=drop_rate,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        src_trg_mask = self.make_pad_mask(trg, src)
        trg_mask = self.make_pad_mask(trg, trg) * self.make_no_peak_mask(trg, trg)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_trg_mask)
        return output

    def make_pad_mask(self, q, k):
        # [batch_size, q_seq_len]
        # [batch_size, k_seq_len]
        len_q, len_k = q.size(1), k.size(1)

        # [batch_size, 1, 1, len_k]
        # torch.ne(input) : computes input != tensor element-wise
        #                   return True where input is not equal to tensor
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # [batch_size, 1, len_q, len_k]
        k = k.repeat(1, 1, len_q, 1)

        # [batch_size, 1, len_q, 1]
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # [batch_size, 1, len_q, len_k]
        q = q.repeat(1, 1, 1, len_k)

        # '&' operation to mask element where the <pad> locate
        mask = k & q
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        # len_q, len_k
        # torch.tril() : The lower triangular part of the matrix
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask
