import torch.nn as nn
from llmplus.utils.clones import clones
from llmplus.layer.Normalize import LayerNorm


class Encoder(nn.Module):
    """
    Encoder which is a stack of N layers
    """
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
