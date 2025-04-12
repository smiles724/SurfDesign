from dataclasses import dataclass
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, SAGEConv


@dataclass
class HeteroGNNConfig:
    D_max: float = 10.0
    num_rbf: int = 16


class HeteroGNN(torch.nn.Module):
    """
    https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous.html
    """
    def __init__(self, in_channels, out_channels, edge_dim=16, num_layers=3):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({   # GCNConv is not allowed for interactions & do not use lazy initialization '-1'
                # ('surface', 'inter', 'protein'): SAGEConv((in_channels, in_channels), in_channels),
                ('surface', 'inter', 'protein'): GATv2Conv(in_channels, in_channels, edge_dim=edge_dim, add_self_loops=False),
            }, aggr='mean')   # todo: aggr: mean or sum?
            self.convs.append(conv)

        self.lin = Linear(in_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        surf_feat = x_dict['surface']
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {'protein': x_dict['protein'].relu(), 'surface': surf_feat}  # TODO: ablation on relu  # surface feature is not updated so it is not included in the output
        return self.lin(x_dict['protein'])
