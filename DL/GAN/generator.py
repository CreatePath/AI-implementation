import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, dim_z, dims_h, out_height, out_width, dropout):
        super().__init__()

        self.dim_z = dim_z
        self.out_height = out_height
        self.out_width = out_width

        self.linear = nn.Linear(dim_z, dims_h[0])
        self.hidden = nn.ModuleList([nn.Sequential(nn.Linear(dims_h[i], dims_h[i+1]),
                                                   nn.Dropout(dropout),
                                                   nn.ReLU(inplace=True)) for i in range(len(dims_h)-1)])
        self.out = nn.Sequential(nn.Linear(dims_h[-1], out_height * out_width),
                                 nn.Sigmoid(),)
    
    def forward(self, z: torch.Tensor):
        out = self.linear(z)
        for m in self.hidden:
            out = m(out)
        out = self.out(out)
        out = out.reshape(-1, 1, self.out_height, self.out_width)
        return out