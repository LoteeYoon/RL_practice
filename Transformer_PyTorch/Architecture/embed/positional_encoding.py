import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    # compute sinusoidal encoding

    def __init__(self, d_model, max_len, device):
        # d_model : dim of model (dim of embedding vectors)
        super().__init__()

        # same size with embedding vector
        self.encoding = torch.zeros(max_len, d_model, device=device)
        # we don't need to compute gradient
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # (max_len, d_model) = [512, 512]
        batch_size, seq_len = x.size()     # (batch_size, seq_len) = [128, 30]
        return self.encoding[:seq_len, :]  # (seq_len, d_model) = [30, 512]
        # it will add with tok_emb : [128, 30, 512]
