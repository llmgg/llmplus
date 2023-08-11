import torch.nn as nn
from llmplus_tmp.utils.clones import clones
from llmplus_tmp.layer.resnet_connection import ResnetConnection
from llmplus_tmp.layer.Normalize import LayerNorm


class EncoderLayer(nn.Module):
    """
    Use self-att -> ff-layer to construct the encoder-layer union
    """

    def __init__(self,
                 d_model: int,
                 self_attn: nn.Module,
                 feed_forward: nn.Module,
                 dropout=0.0
                 ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm = clones(LayerNorm(d_model), 2)
        self.sublayer = clones(ResnetConnection(dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        # 1) self-att
        x = self.sublayer[0](self.norm[0](x), lambda x: self.self_attn(x, x, x, mask))
        # 2) ff layer
        return self.sublayer[1](self.norm[1](x), self.feed_forward)
