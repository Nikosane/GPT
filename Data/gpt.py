import os
import torch
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200
batch_size = 32
block_size = 64
max_iters = 15000
eval_interval = 1000
learning_rate = 1e-3
n_embd = 256
n_head = 4
n_layer = 8
dropout = 0.2