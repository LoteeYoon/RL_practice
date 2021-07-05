import math
import torch.nn as nn


class ScaleDotProductAttention(nn.Module):
    # encoder's Attention : Query = Key = Value
    # decoder's first Attention : Query = Key = Value
    # decoder's second Attention : Query : decoder's vector / Key = Value : encoder's vector
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input : [batch_size, n_head, seq_len, d_k]
        # seq_len : seq_len of the sentence of q, k, v
        # d_k : d_model / n_head
        batch_size, n_head, seq_len, d_k = k.size()

        # 1. dot product q with k^T (similarity)
        k_t = k.view(batch_size, n_head, d_k, seq_len)
        score = (q @ k_t) / math.sqrt(d_k)

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with value
        out = score @ v

        return out, score
