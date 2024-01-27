import os
import torch
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'