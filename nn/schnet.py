import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from nn.schnet_conv import SchnetConv
from nn.ssp import ShiftedSoftplus

pooler_dict = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
    None: None,
}


class SchNet(nn.Module):
    """SchNet model for geometric graphs [https://arxiv.org/pdf/1706.08566.pdf]"""

    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        depth,
        weight_net_dims=[64],
        num_basis=300,
        d_min=0,
        d_max=30,
        gamma=10,
        act=ShiftedSoftplus,
        aggr="add",
        pooler="add",
    ):
        super().__init__()

        self.embedder = nn.Linear(in_features, hidden_features)

        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(
                SchnetConv(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    weight_net_dims=weight_net_dims,
                    num_basis=num_basis,
                    d_min=d_min,
                    d_max=d_max,
                    gamma=gamma,
                    act=act,
                    aggr=aggr,
                )
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            act(),
            nn.Linear(hidden_features, out_features),
        )
        self.pooler = pooler_dict[pooler]

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        dist = graph.edge_attr
        batch = graph.batch

        x = self.embedder(x)
        for conv in self.convs:
            x = x + conv(x, edge_index, dist)  # residual connection

        x = self.head(x)
        if self.pooler is not None:
            x = self.pooler(x, batch)
        return x
