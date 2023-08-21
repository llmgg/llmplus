import torch
import torch.nn as nn


def calculate_att_score(
        query: torch.Tensor,
        key: torch.Tensor,
        nhead: int
) -> torch.Tensor:
    """
    Calculate the attention scores between query and key

    The size of input tensor:
    query: (batch_size, nq, nhead*q_dim)
    key: (batch_size, nk, nhead*k_dim)

    The size of output tensor:
    att_score: (batch_size, nhead, nq, nk)
    """
    assert query.size(-1) == key.size(-1), \
        "query.dim({}) != key.dim({})".format(query.size(-1), key.size(-1))

    batch_size, nq, d_model = query.size()
    assert d_model % nhead == 0, \
        "the head is {}, but the d_model is {}.".format(nhead, d_model)

    # (batch_size*nhead, nq, q_dim)
    q_dim = d_model // nhead
    query = query.contiguous().view(batch_size, nq, nhead, q_dim).\
        transpose(1, 2).reshape(batch_size * nhead, nq, q_dim)
    query *= q_dim ** -0.5
    # (batch_size*nhead, nk, k_dim)
    key = key.contiguous().view(batch_size, -1, nhead, q_dim).\
        transpose(1, 2).reshape(batch_size*nhead, -1, q_dim)

    # (batch_size, nhead, nq, nk)
    return torch.bmm(query, key.mT).contiguous().\
        view(batch_size, nhead, nq, -1)


def calculate_att_value(
        att_score: torch.Tensor,
        value: torch.Tensor,
        nhead: int
) -> torch.Tensor:
    """
    Calculate the attention value between att_score and value

    The size of input tensor:
    att_score: (batch_size, nhead, nq, nk)
    value: (batch_size, nv, nhead*v_dim)

    The size of output tensor:
    att_value: (batch_size, nq, nhead*v_dim)
    """
    batch_size, nv, dv = value.size()
    assert att_score.size(-1) == nv, \
        "Tokens in key({}) != Tokens in Value({})".format(
            att_score.size(-1), nv
        )
    assert dv % nhead == 0, \
        "the head is {}, but the dv is {}.".format(nhead, dv)
    # (batch_size*nhead, nv, v_dim)
    v_dim = dv // nhead
    value = value.contiguous().view(batch_size, nv, nhead, v_dim).\
        transpose(1, 2).reshape(batch_size * nhead, nv, v_dim)

    # (batch_size*nhead, nq, nk)
    att_score = att_score.contiguous().view(batch_size * nhead, -1, nv)

    # (batch_size, nq, nhead*v_dim)
    return torch.bmm(att_score, value).contiguous().\
        view(batch_size, nhead, -1, v_dim).transpose(1, 2).\
        reshape(batch_size, -1, nhead * v_dim)


def attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        nhead: int, mask=None, dropout=None
) -> torch.Tensor:
    """
    The original token-wise dot product attention layer used in self-att_score
    and src-att_score

    The size of input tensor:
    query: (batch_size, nq, nhead*q_dim)
    key: (batch_size, nk, nhead*k_dim)
    value: (batch_size, nv, nhead*v_dim)

    Note:
        1. q_dim == k_dim
        2. nk == nv

    The size of output tensor:
    (batch_size, nq, nhead*v_dim)
    """
    att_score = calculate_att_score(query, key, nhead)

    if mask is not None:
        att_score = att_score.masked_fill(mask == 0, -1.0e9)
    att_score = att_score.softmax(dim=-1)
    if dropout is not None:
        att_score = dropout(att_score)

    # (batch_size, nq, nhead*v_dim)
    return calculate_att_value(att_score, value, nhead)


def swa(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        nhead: int, mask=None, dropout=None
) -> torch.Tensor:
    """
    Implementation of sentence-wise attention.

    The size of input tensor:
    query: (batch_size, nq, nhead*q_dim)
    key: (batch_size, nk, nhead*k_dim)
    value: (batch_size, nv, nhead*v_dim)

    Note:
        1. q_dim == k_dim
        2. nk == nv

    The size of output tensor:
    (batch_size, nq, nhead*v_dim)
    """
    pass


class MultiHeadedAttention(nn.Module):
    """
    Multi Head Attention Layer
    """
    def __init__(self, nhead: int, d_model: int, dropout=0.1, v_dim=None):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % nhead == 0, \
            "The input nhead is {}, but d_model is {}".format(nhead, d_model)

        self.nhead = nhead
        self.q_dim = d_model // nhead
        self.k_dim = self.q_dim
        self.v_dim = self.q_dim if v_dim is None else v_dim
        self.q_layer = nn.Linear(d_model, self.nhead * self.q_dim)
        self.k_layer = nn.Linear(d_model, self.nhead * self.k_dim)
        self.v_layer = nn.Linear(d_model, self.nhead * self.v_dim)
        self.o_layer = nn.Linear(self.nhead * self.v_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask=None):
        """
        The size of input tensor:
        query: (batch_size, nq, d_model)
        key: (batch_size, nk, d_model)
        value: (batch_size, nv, nhead*v_dim)

        In self-att, query, key and value is from a same tensor.
        In src-att, query is the output of decoder, while key and value is the ones
        of encoder.

        The size of output tensor:
        (batch_size, nq, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size, nq, d_model = query.size()

        # 1) calculate q, k and v with linear model
        [query, key, value] = [
            layer(data).view(batch_size, -1, self.nhead * dim)
            for (layer, data, dim) in zip(
                (self.q_layer, self.k_layer, self.v_layer),
                (query, key, value),
                (self.q_dim, self.k_dim, self.v_dim)
            )
        ]

        # 2) calculate the attention value
        # (batch_size, nq, nhead * v_dim)
        x = attention(
            query, key, value, self.nhead, mask, self.dropout
        )
        del query
        del key
        del value

        # 3) concat the value and output the result
        # (batch_size, nq, d_model)
        return self.o_layer(x)
