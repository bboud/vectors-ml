from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.to(torch.cfloat)
        # Input Layer
        self.linear_in = nn.Linear(64, 32, bias=False, dtype=torch.cfloat)

        self.linear_1 = nn.Linear(32, 32, bias=False, dtype=torch.cfloat)

        self.linear_2 = nn.Linear(32, 141, bias=False, dtype=torch.cfloat)

        self.tanh = nn.Tanh()

    def forward(self, t):
        t = self.linear_in(t)

        t = self.tanh(self.linear_1(t))

        t = self.tanh(self.linear_2(t))

        return t