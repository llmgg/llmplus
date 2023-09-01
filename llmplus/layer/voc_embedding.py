import math
import torch
import torch.nn as nn


class VocEmbedding(nn.Module):
    def __init__(self, voc_size: int, dim: int, padding_idx = None):
        super(VocEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=dim,
            padding_idx=padding_idx,
        )
        self.coe = math.sqrt(dim)

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * self.coe
