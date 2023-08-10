import math

import torch
import torch.nn as nn


class VocEmbedding(nn.Module):
    def __init__(self, voc_size: int, dim: int):
        super(VocEmbedding, self).__init__()
        self.embedding = nn.Embedding(voc_size, dim)
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.dim)
