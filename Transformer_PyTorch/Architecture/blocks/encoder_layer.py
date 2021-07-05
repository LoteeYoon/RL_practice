import torch.nn as nn
from Architecture.layers.layer_norm import LayerNorm
from Architecture.layers.position_wise_feed_forward import PositionWiseFeedForward
from Architecture.layers.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_head, drop_rate):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_rate)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop_rate=drop_rate)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_rate)

    def forward(self, x, s_mask):
        # 1. self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x

