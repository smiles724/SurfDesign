from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_geometric
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from torch.nn import SiLU, Linear, Sequential, Sigmoid, ReLU, Identity, Dropout
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, OptTensor, Tensor

from .utils import *

_pyg_version_ = '2.3.1'


@dataclass
class EGNNConfig:
    n_layers: int = 3
    in_dim: int = 1
    feat_dim: int = 16
    m_dim: int = 16
    edge_attr_dim: int = 16
    cutoff: float = 5.0
    num_spherical: int = 7
    num_radial: int = 6
    dropout: float = 0.15
    update_coors: bool = False
    norm_coors: bool = True
    coor_weights_clamp_value: float = 2.0


# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=SiLU()):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


############################################
# Helper function in DiG Dimenet ++ & SchNet
############################################

class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    """ distance embedding in EGNN
        Usage: fourier_encode_dist(rel_dist)  # (*, 2 * fourier_features + 1)
        """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x.squeeze(1)


class dist_emb_v1(torch.nn.Module):
    """ Gaussian distance embedding in SchNet """

    def __init__(self, start=0.0, stop=5.0, num_gaussians=10):
        super(dist_emb_v1, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class dist_emb_v2(torch.nn.Module):
    """ distance embedding in DimeNet++ """

    def __init__(self, num_radial=5, cutoff=5.0, envelope_exponent=5):
        super(dist_emb_v2, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(math.pi)

    def forward(self, dist):
        dist /= self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class angle_emb(torch.nn.Module):
    """ angle embedding in DimeNet++ """

    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5):
        super(angle_emb, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, src_angle, dst_angle, dih_angle):
        n, k = self.num_spherical, self.num_radial
        dist = dist.squeeze(-1) / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)  # (*, num_spherical × num_radial)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = [torch.stack([f(a) for f in self.sph_funcs], dim=1) for a in [src_angle, dst_angle, dih_angle]]  # list of (*, num_spherical, 1)
        out = torch.cat([rbf.view(-1, n, k) * i.view(-1, n, 1) for i in cbf], dim=-1)
        return out.view(-1, n * k * 3)


def compute_angle(pos_ji, pos_jk):
    """
    Calculate angles. 0 to pi
        pos_ij: vector pointing from j to i
        pos_jk: vector pointing from j to k
    """
    a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
    b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
    angle = torch.atan2(b, a)
    return angle


class update_e(torch.nn.Module):
    def __init__(self, hidden_dim, int_emb_size, basis_emb_size, num_spherical, num_radial, num_before_skip=1, num_after_skip=2, act=SiLU()):
        super(update_e, self).__init__()
        self.act = act
        self.lin_rbf0 = nn.Sequential(nn.Linear(num_radial, basis_emb_size, bias=False), nn.ReLU(), nn.Linear(basis_emb_size, hidden_dim, bias=False))
        self.lin_sbf0 = nn.Sequential(nn.Linear(num_spherical * num_radial * 3, basis_emb_size, bias=False), nn.ReLU(), nn.Linear(basis_emb_size, int_emb_size, bias=False))
        self.lin_rbf = nn.Linear(num_radial, hidden_dim, bias=False)

        self.lin_ij = nn.Linear(hidden_dim, hidden_dim)

        self.lin_down = nn.Linear(hidden_dim, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_dim, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([ResidualLayer(hidden_dim, act) for _ in range(num_before_skip)])
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.layers_after_skip = torch.nn.ModuleList([ResidualLayer(hidden_dim, act) for _ in range(num_after_skip)])
        self.reset_parameters()

    def reset_parameters(self):
        for i in [0, 2]:  # skip act
            self.lin_rbf0[i].reset_parameters()
            self.lin_sbf0[i].reset_parameters()
        glorot_orthogonal(self.lin_ij.weight, scale=2.0)
        self.lin_ij.bias.data.fill_(0)
        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, e, rbf0, sbf):
        e0, _ = e
        x_ij = self.act(self.lin_ij(e0))
        x_ij = x_ij * self.lin_rbf0(rbf0)
        x_ij = self.act(self.lin_down(x_ij))
        x_ij = x_ij * self.lin_sbf0(sbf)
        e1 = self.act(self.lin_up(x_ij))

        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + e0
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1
        return e1, e2


############################################
# Equivariant GNN
############################################

class EGNN_Layer(MessagePassing):
    """ Different from the above since it separates the edge assignment
        from the computation (this allows for great reduction in time and 
        computations when the graph is locally or sparse connected).
        Args:
            aggr: one of ["add", "mean", "max"]
    """

    def __init__(self, feat_dim, edge_attr_dim=0, m_dim=16, soft_edge=0, norm_feats=False, norm_coors=False, norm_coors_scale_init=1e-2, update_coors=True, dropout=0., aggr="add",
                 edge_e3nn=None, **kwargs):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method must be a valid option'
        kwargs.setdefault('aggr', aggr)
        super(EGNN_Layer, self).__init__(**kwargs)
        self.update_coors = update_coors
        self.edge_dim = edge_attr_dim + feat_dim * 2
        self.dropout = Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        self.edge_weight = Sequential(Linear(m_dim, 1), Sigmoid()) if soft_edge else None
        self.edge_mlp = Sequential(Linear(self.edge_dim, self.edge_dim * 2), self.dropout, SiLU(), Linear(self.edge_dim * 2, m_dim), SiLU())
        self.edge_e3nn = edge_e3nn
        if self.edge_e3nn:
            self.irreps_edge_attr, self.max_radius, self.num_neighbors = edge_e3nn['irreps_edge_attr'], edge_e3nn['max_radius'], edge_e3nn['num_neighbors']
            irreps_in, self.num_basis, radial_layers, radial_neurons = edge_e3nn['irreps_in'], edge_e3nn['num_basis'], edge_e3nn['radial_layers'], edge_e3nn['radial_neurons']
            irreps_out = irreps_in
            irreps_mid = []
            instructions = []
            for i, (mul, ir_in) in enumerate(irreps_in):
                for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in irreps_out:
                            k = len(irreps_mid)
                            irreps_mid.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))
            irreps_mid = o3.Irreps(irreps_mid)
            irreps_mid, p, _ = irreps_mid.sort()

            self.tp = TensorProduct(irreps_in, self.irreps_edge_attr, irreps_mid, instructions, internal_weights=False, shared_weights=False, )
            self.fc = FullyConnectedNet([self.num_basis] + radial_layers * [radial_neurons] + [self.tp.weight_numel], nn.functional.silu)
            self.lin2 = FullyConnectedTensorProduct(irreps_mid, irreps_in, irreps_out)

        # NODES - can't do identity in node_norm bc pyg expects 2 inputs, but identity expects 1.
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feat_dim) if norm_feats else Identity()
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else Identity()
        self.node_mlp = Sequential(Linear(feat_dim + m_dim, feat_dim * 2), self.dropout, SiLU(), Linear(feat_dim * 2, feat_dim))
        self.coors_mlp = Sequential(Linear(m_dim, m_dim * 4), self.dropout, SiLU(), Linear(m_dim * 4, 1)) if update_coors else None

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """ propagating messages. """
        if _pyg_version_ == '2.3.1':
            size = self._check_input(edge_index, size)
            coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        else:
            size = self.__check_input__(edge_index, size)
            coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        m_ij = self.message(**msg_kwargs)  # get messages
        coors_out = kwargs["coord"]
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            rel_coors = self.coors_norm(kwargs["rel_coors"])  # normalize if needed
            mhat_i = self.aggregate(coor_wij * rel_coors, **aggr_kwargs)
            coors_out += mhat_i
        if self.edge_weight:  # weight the edges
            m_ij = m_ij * self.edge_weight(m_ij)
        m_i = self.aggregate(m_ij, **aggr_kwargs)

        hidden_feats = self.node_norm(kwargs["x"])
        hidden_out = kwargs["x"] + self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        return self.update((hidden_out, coors_out), **update_kwargs)

    def forward(self, coord, feats, edge_index: Adj, edge_attr: OptTensor = None, batch: Adj = None, ) -> Tensor:
        """ Inputs:
            * edge_index: (2, n_edges)
            * edge_attr: tensor (n_edges, n_feats) excluding basic distance feats.
        """
        edge_src, edge_dst = edge_index[0], edge_index[1]
        rel_coors = coord[edge_src] - coord[edge_dst]
        if self.edge_e3nn:  # update the edge attr by e3nn
            edge_vec = coord[edge_src] - coord[edge_dst]
            edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization="component")
            edge_length = edge_vec.norm(dim=1)
            edge_length_embedded = soft_one_hot_linspace(x=edge_length, start=0.0, end=self.max_radius, number=self.num_basis, basis="gaussian", cutoff=False).mul(
                self.num_basis ** 0.5)
            edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

            weight = self.fc(edge_length_embedded)
            edge_features = self.tp(feats[edge_src], edge_attr, weight)
            x = scatter(edge_features, edge_dst, dim_size=feats.shape[0]).div(self.num_neighbors ** 0.5)
            feats = self.lin2(x, feats)

        hidden_out, coors_out = self.propagate(edge_index, x=feats, edge_attr=edge_attr, coord=coord, rel_coors=rel_coors, batch=batch)
        return coors_out, hidden_out


class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_dim, act=SiLU()):
        super(init, self).__init__()
        self.act = act
        self.lin_rbf_0 = Linear(num_radial, hidden_dim)
        self.lin = Linear(3 * hidden_dim, hidden_dim)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, dis_emb, edge_index):
        rbf0 = self.act(self.lin_rbf_0(dis_emb))
        e1 = self.act(self.lin(torch.cat([x[edge_index[0]], x[edge_index[1]], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(dis_emb) * e1
        return e1, e2


class EGNN_Network(nn.Module):
    r"""Sample GNN model architecture that uses the Backbone-Sparse message passing layer to learn over point clouds.
        Main MPNN layer introduced in https://arxiv.org/abs/2102.09844v1
        Args:
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            envelope_exponent (int, optional): Shape of the smooth cutoff. (default: :obj:`5`)
            basis_emb_size (int, optional): Embedding size used in the basis transformation. (default: :obj:`8`)
    """

    def __init__(self, n_layers, in_dim, feat_dim, edge_attr_dim=16, m_dim=16, soft_edge=0, cutoff=5.0, num_spherical=7, num_radial=6, envelope_exponent=5,
                 basis_emb_size=8, update_coors=True, norm_feats=True, norm_coors=False, norm_coors_scale_init=1e-2, dropout=0., coor_weights_clamp_value=None, edge_e3nn=None):
        super().__init__()
        self.in_dim = in_dim
        self.update_coors = update_coors
        self.dist_emb = dist_emb_v2(num_radial, cutoff, envelope_exponent)
        self.angle_emb = angle_emb(num_spherical, num_radial, cutoff, envelope_exponent)

        self.init_v = Sequential(Linear(in_dim, feat_dim // 2), ReLU(), Linear(feat_dim // 2, feat_dim))
        self.init_e = init(num_radial, edge_attr_dim)

        self.update_e = nn.ModuleList([update_e(edge_attr_dim, edge_attr_dim // 2, basis_emb_size, num_spherical, num_radial) for _ in range(n_layers)])
        # self.mpnn_layers = nn.ModuleList([EGNN_Layer(feat_dim=feat_dim, edge_attr_dim=edge_attr_dim, m_dim=m_dim, soft_edge=soft_edge, norm_feats=norm_feats,
        #                                              norm_coors=norm_coors, norm_coors_scale_init=norm_coors_scale_init, update_coors=update_coors, dropout=dropout,
        #                                              coor_weights_clamp_value=coor_weights_clamp_value, edge_e3nn=edge_e3nn) for _ in range(n_layers)])
        self.mpnn_layers = nn.ModuleList([EGNN_Layer(feat_dim=feat_dim, edge_attr_dim=edge_attr_dim, m_dim=m_dim, soft_edge=soft_edge, norm_feats=norm_feats,
                                                     norm_coors=norm_coors, norm_coors_scale_init=norm_coors_scale_init, update_coors=update_coors, dropout=dropout,
                                                     edge_e3nn=edge_e3nn) for _ in range(n_layers)])

    def forward(self, surf_batch):
        coord, norm, edge_index, batch = surf_batch.pos, surf_batch.norm, surf_batch.edge_index, surf_batch.batch
        feat = torch.stack([surf_batch.hp, surf_batch.hbond], dim=-1)

        edge_src, edge_dst = edge_index[0], edge_index[1]
        rel_coors = coord[edge_src] - coord[edge_dst]
        rel_dist = rel_coors.pow(2).sum(dim=-1, keepdim=True).sqrt()
        dist_emb = self.dist_emb(rel_dist)  # (*, num_radial)

        # TODO: add angles (loss becomes nan? or val_acc = 0?)
        src_angle = compute_angle(norm[edge_src], -rel_coors)  # angles between src's norm and direction
        dst_angle = compute_angle(norm[edge_dst], rel_coors)  # angle between dst's norm and direction
        dih_angle = compute_angle(norm[edge_src], norm[edge_dst])  # dihedral angles
        angle_emb = self.angle_emb(rel_dist, src_angle, dst_angle, dih_angle)  # (*, 3 × num_spherical × num_radial), may have infinity or nan values

        feat = self.init_v(feat)
        edge_attr = surf_batch.get('edge_attr', None)
        edge_attr = self.init_e(feat, dist_emb, edge_index) if edge_attr is None else edge_attr

        for i, layer in enumerate(self.mpnn_layers):
            # update coords, node attr, edge attr
            edge_attr = self.update_e[i](edge_attr, dist_emb, angle_emb)
            coord, feat = layer(coord, feat, edge_index, edge_attr[1], batch=batch)

            # update distance and angles for new coords
            if self.update_coors and i != len(self.mpnn_layers) - 1:
                rel_coors = (coord[edge_src] - coord[edge_dst]).detach()  # no grad
                rel_dist = rel_coors.pow(2).sum(dim=-1, keepdim=True).sqrt()
                dist_emb = self.dist_emb(rel_dist)

                src_angle = compute_angle(norm[edge_src], -rel_coors)
                dst_angle = compute_angle(norm[edge_dst], rel_coors)
                angle_emb = self.angle_emb(rel_dist, src_angle, dst_angle, dih_angle)

        return feat

    def __repr__(self):
        return 'EGNN_Network of: {0} layers'.format(len(self.mpnn_layers))
