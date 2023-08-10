import torch.nn as nn
import copy
from llmplus.layer.voc_embedding import VocEmbedding
from llmplus.layer.positional_embedding import PositionalEmbedding
from llmplus.layer.attention import MultiHeadedAttention
from llmplus.coder.encoder import Encoder
from llmplus.coder.encoder_layer import EncoderLayer
from llmplus.coder.decoder import Decoder
from llmplus.coder.decoder_layer import DecoderLayer
from llmplus.layer.feed_forward import FeedForward
from llmplus.generator.generator import Generator
from llmplus.models.encoder_decoder import EncoderDecoder


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(nhead=h, d_model=d_model, dropout=dropout)
    ff = FeedForward(d_input=d_model, d_hidden=d_ff, dropout=dropout)
    position = PositionalEmbedding(d_model=d_model, dropout=dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(VocEmbedding(src_vocab, d_model), c(position)),
        nn.Sequential(VocEmbedding(tgt_vocab, d_model), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
