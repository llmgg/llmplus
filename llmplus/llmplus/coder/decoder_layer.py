import torch.nn as nn
from llmplus.utils.clones import clones
from llmplus.layer.resnet_connection import ResnetConnection
from llmplus.layer.Normalize import LayerNorm


class DecoderLayer(nn.Module):
    """
    Use self-att -> src-att -> ff-layer to construct the decoder-layer union
    """

    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm = clones(LayerNorm(d_model), 3)
        self.sublayer = clones(ResnetConnection(dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 1) self-att
        x = self.sublayer[0](self.norm[0](x), lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2) src-att
        m = memory
        x = self.sublayer[1](self.norm[1](x), lambda x: self.src_attn(x, m, m, src_mask))
        # 3) ff layer
        return self.sublayer[2](self.norm[2](x), self.feed_forward)
