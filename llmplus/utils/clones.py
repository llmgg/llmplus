import torch.nn as nn
import copy


def clones(module, N):
    """
    Construct N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
