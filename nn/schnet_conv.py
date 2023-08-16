import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from nn.gaussian_basis import GaussianBasis


class SchnetConv(MessagePassing):
    """SchNet convolution module [https://arxiv.org/pdf/1706.08566.pdf], slightly customisable"""

    def __init__(
        self,
        in_features,
        out_features,
        weight_net_dims,
        num_basis,
        d_min,
        d_max,
        gamma,
        aggr,
        act,
    ):
        super().__init__(aggr=aggr)

        self.atomwise1 = nn.Linear(in_features, in_features)
        self.atomwise2 = nn.Sequential(
            nn.Linear(in_features, in_features),
            act(),
            nn.Linear(in_features, out_features),
        )

        mlp = [GaussianBasis(num_basis, d_min, d_max, gamma)]
        dims = [num_basis] + weight_net_dims + [in_features]

        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i + 1]))
            mlp.append(act())

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x, edge_index, dist):
        x = self.atomwise1(x)
        x = self.propagate(edge_index, x=x, dist=dist)
        x = self.atomwise2(x)
        return x

    def message(self, x_j, dist):
        W = self.mlp(dist)
        return x_j * W

    def __repr__(self):
        return "{}(\n atomwise1:{} \n weight net:{} \n atomwise2:{} \n)".format(
            self.__class__.__name__,
            self.atomwise1,
            self.mlp,
            self.atomwise2,
        )
