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

with open('/content/university data','r', encoding='utf-8') as f:
    text = f.read()
text = ' '.join(text.split())

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)- block_size , (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device) , y.to(device)
    return x, y

@torch.no_grad()
def extimate_loss():
    out = {}
    model.eval()
    for split in ['train' , 'val']:
        losses = torch.zeros(eval_iters)
        