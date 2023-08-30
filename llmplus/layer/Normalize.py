import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    """
    Construct a normalization layer with Layer Norm method.
    """
    def __init__(self, feature_size, eps=1.0e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feature_size))
        self.beta = nn.Parameter(torch.zeros(feature_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (torch.sqrt(var) + self.eps)
        return self.gamma * x + self.beta
