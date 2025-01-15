import torch
from torch import nn

def make_hidden_layer(in_dim, out_dim, activation=nn.ReLU, dropout=0.3):
    return nn.Sequential(nn.Linear(in_dim, out_dim),
                         nn.BatchNorm1d(out_dim),
                         activation(),
                         nn.Dropout(dropout),)

class Generator(nn.Module):
    def __init__(self, dim_z, dim_y, out_dim, dims_h: list[int], activation=nn.ReLU, dropout=0.3):
        super().__init__()

        n_layers = len(dims_h) - 1
        hidden_z = dims_h[0] // 2
        hidden_y = dims_h[0] - hidden_z

        self.linear_z = nn.Sequential(nn.Linear(dim_z, hidden_z),
                                      nn.BatchNorm1d(hidden_z),
                                      activation(),
                                      nn.Linear(hidden_z, hidden_z),
                                      nn.BatchNorm1d(hidden_z),
                                      activation(),)

        self.linear_y = nn.Sequential(nn.Linear(dim_y, hidden_y),
                                      nn.BatchNorm1d(hidden_y),
                                      activation(),
                                      nn.Linear(hidden_y, hidden_y),
                                      nn.BatchNorm1d(hidden_y),
                                      activation(),)

        self.hidden_layers = nn.ModuleList([make_hidden_layer(dims_h[i],
                                                              dims_h[i+1],
                                                              activation,
                                                              dropout=dropout) for i in range(n_layers)])
        self.final_layer = nn.Sequential(nn.Linear(dims_h[-1], out_dim),
                                         nn.Sigmoid(),)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        z = self.linear_z(z)
        y = self.linear_y(y)
        x = torch.cat([z, y], dim=1)
        for h_layer in self.hidden_layers:
            x = h_layer(x)
        x = self.final_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, dims_h, activation=nn.ReLU, dropout=0.3):
        super().__init__()

        n_layers = len(dims_h) - 1
        self.input_layer = nn.Sequential(nn.Linear(in_dim, dims_h[0]),
                                         nn.BatchNorm1d(dims_h[0]),
                                         activation(),)
        self.hidden_layers = nn.ModuleList([make_hidden_layer(dims_h[i],
                                                              dims_h[i+1],
                                                              activation,
                                                              dropout) for i in range(n_layers)])
        self.final_layer = nn.Sequential(nn.Linear(dims_h[-1], out_dim),
                                         nn.Sigmoid(),)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.input_layer(x)
        for h_layer in self.hidden_layers:
            x = h_layer(x)
        x = self.final_layer(x)

        return x