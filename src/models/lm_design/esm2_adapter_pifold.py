from dataclasses import dataclass, field
from typing import List
from torch_geometric.utils import to_dense_batch

import torch
from torch.nn import functional as F
from src.models import register_model
from src.models._base import FixedBackboneDesignEncoderDecoder
from src.models.generator import sample_from_categorical
# from src.models.protein_mpnn_cmlm.protein_mpnn import (ProteinMPNNCMLM, ProteinMPNNConfig)
from src.models.pifold import PiFold, PiFoldConfig
from src.models.surf_design.egnn import (EGNN_Network, EGNNConfig)
from src.models.surf_design.heteroGNN import HeteroGNN
from src.models.pifold.modules import _rbf

from .modules.esm2_adapter import ESM2WithStructuralAdatper

@dataclass
class ESM2AdapterConfig:
    encoder: PiFoldConfig = field(default=PiFoldConfig())
    surf_encoder: EGNNConfig = field(default=EGNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [32, ])
    separate_loss: bool = True
    name: str = 'esm2_t33_650M_UR50D'
    dropout: float = 0.1


@register_model('esm2_adapter_pifold')
class ESM2AdapterPiFold(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ESM2AdapterConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.surf_encoder = EGNN_Network(**self.cfg.surf_encoder)
        self.interact = HeteroGNN(in_channels=self.cfg.surf_encoder.feat_dim, out_channels=self.cfg.encoder.d_model, num_layers=3)  # TODO: a separate config

        self.encoder = PiFold(self.cfg.encoder)
        self.decoder = ESM2WithStructuralAdatper.from_pretrained(args=self.cfg, name=self.cfg.name)

        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx

    def forward_surface(self, batch):
        surfs = batch['prot_surfs']['surface']
        proteins = batch['prot_surfs']['protein']
        edge_index = batch['prot_surfs']['surface', 'inter', 'protein'].edge_index

        feat = torch.stack([surfs.hp, surfs.hbond], dim=-1)
        encoder_out_ = self.surf_encoder(coord=surfs.pos, feat=feat, edge_index=surfs.edge_index, batch=surfs.batch, )

        x_dict = {'protein': torch.zeros(proteins.num_nodes, encoder_out_.shape[-1]).to(encoder_out_.device), 'surface': encoder_out_}
        edge_index_dict = {('surface', 'inter', 'protein'): edge_index}
        edge_attr = _rbf(torch.norm(surfs.pos[edge_index[0]] - proteins.pos[edge_index[1]], dim=-1), num_rbf=16)  # TODO: num_rbf -> hyper # TODO: change D_max
        edge_attr_dict = {('surface', 'inter', 'protein'): edge_attr}
        interact_out = self.interact(x_dict, edge_index_dict, edge_attr_dict)

        interact_out, mask = to_dense_batch(interact_out, proteins.batch)  # not matter mask not the same as batch['coord_mask']
        interact_out = F.pad(interact_out, (0, 0, 1, 1), value=0)  # todo: not padding when coord_pad_inf is False
        return interact_out

    def forward(self, batch, **kwargs):
        interact_out = self.forward_surface(batch)
        encoder_logits, encoder_out = self.encoder(batch, init_feat=interact_out, return_feats=True, **kwargs)
        encoder_out['feats'] = encoder_out['feats'].detach()    # detach, no gradient required
        init_pred = encoder_logits.argmax(-1)
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])
        esm_logits = self.decoder(tokens=init_pred, encoder_out=encoder_out)['logits']

        if not getattr(self.cfg, 'separate_loss', False):
            logits = encoder_logits + esm_logits     # a combined logit, TODO: use only the final logits
            return logits, encoder_logits

        return esm_logits, encoder_logits

    def forward_encoder(self, batch):
        interact_out = self.forward_surface(batch)
        encoder_logits, encoder_out = self.encoder(batch, init_feat=interact_out, return_feats=True)

        init_pred = encoder_logits.argmax(-1)
        init_pred = torch.where(batch['coord_mask'], init_pred, batch['prev_tokens'])

        encoder_out['logits'] = encoder_logits
        encoder_out['init_pred'] = init_pred
        encoder_out['coord_mask'] = batch['coord_mask']
        return encoder_out

    def forward_decoder(self, prev_decoder_out, encoder_out, need_attn_weights=False):
        output_tokens = prev_decoder_out['output_tokens']
        output_scores = prev_decoder_out['output_scores']
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = output_tokens.ne(self.padding_idx)  # & coord_mask

        esm_out = self.decoder(tokens=output_tokens, encoder_out=encoder_out)
        esm_logits = esm_out['logits']

        if not getattr(self.cfg, 'separate_loss', False):
            logits = esm_logits + encoder_out['logits']
        else:
            logits = esm_logits

        _tokens, _scores = sample_from_categorical(logits, temperature=temperature)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        history.append(output_tokens.clone())

        return dict(output_tokens=output_tokens, output_scores=output_scores, attentions=None,  # [B, L, H, T, T]
                    step=step + 1, max_step=max_step, history=history, )

    def initialize_output_tokens(self, batch, encoder_out):
        prev_tokens = batch['prev_tokens']
        prev_token_mask = batch['prev_token_mask']

        initial_output_tokens = torch.where(prev_token_mask, encoder_out['init_pred'], prev_tokens)
        initial_output_scores = torch.zeros(*initial_output_tokens.size(), device=initial_output_tokens.device)

        return initial_output_tokens, initial_output_scores
