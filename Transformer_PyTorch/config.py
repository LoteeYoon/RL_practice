import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# model parameter setting
batch_size = 128
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
d_ff = 2048
drop_rate = 0.1

# optimizer parameter setting
initial_lr = 1e-4
lr_decay = 0.9
adam_eps = 5e-8
patience = 10
warmup = 100
epoch = 10
clip = 1.0
weight_decay = 5e-4
inf = float('inf')