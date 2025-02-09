import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, dim_x, dims_h, dim_out, dropout):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(dim_x, dims_h[0]),
                                    nn.Dropout(dropout),
                                    nn.ReLU(inplace=True),)
        self.hidden = nn.ModuleList([nn.Sequential(nn.Linear(dims_h[i], dims_h[i+1]),
                                                   nn.Dropout(dropout),
                                                   nn.ReLU(inplace=True)) for i in range(len(dims_h)-1)])
        self.out = nn.Sequential(nn.Linear(dims_h[-1], dim_out),
                                 nn.Sigmoid(),)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.linear(x)
        for m in self.hidden:
            out = m(out)
        out = self.out(out)
        return out