import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed Forward Layer
    """

    def __init__(self, d_input: int, d_hidden: int, dropout=0.1):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_input)
        )

    def forward(self, x):
        return self.layer(x)
