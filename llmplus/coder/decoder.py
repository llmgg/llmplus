import torch.nn as nn
from llmplus.utils.clones import clones
from llmplus.layer.Normalize import LayerNorm


class Decoder(nn.Module):
    """
    Decoder which is a stack of N layers
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
