import torch.nn as nn
from Architecture.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # input q, k, v is embedded vector that have [batch_size, seq_len, d_model] dimension
        # 1. generate q, k, v
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. compute similarity (scale dot product)
        out, score = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        # tensor : [batch_size, seq_len, d_model]
        # return : [batch_size, n_head, seq_len, d_k]
        batch_size, seq_len, d_model = tensor.size()
        d_k = d_model // self.n_head
        tensor = tensor.view(batch_size, self.n_head, seq_len, d_k)
        return tensor

    def concat(self, tensor):
        # tensor : [batch_size, n_head, seq_len, d_k]
        # return : [batch_size, seq_len, d_model]
        batch_size, n_head, seq_len, d_k = tensor.size()
        d_model = n_head * d_k
        tensor = tensor.view(batch_size, seq_len, d_model)
        return tensor
