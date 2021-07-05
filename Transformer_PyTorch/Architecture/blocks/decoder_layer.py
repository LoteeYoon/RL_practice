import torch.nn as nn
from Architecture.layers.layer_norm import LayerNorm
from Architecture.layers.multi_head_attention import MultiHeadAttention
from Architecture.layers.position_wise_feed_forward import PositionWiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, n_head, drop_rate):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_rate)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_rate)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop_rate=drop_rate)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_rate)

    def forward(self, dec, enc, t_mask, s_mask):
        # 1. self attention
        _x = dec
        x = self.attention(q=dec, k=dec, v=dec, mask=t_mask)

        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            # 3. compute encoder-decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=s_mask)

            # 4. add and norm
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.norm3(x + _x)
        x = self.dropout3(x)
        return x
