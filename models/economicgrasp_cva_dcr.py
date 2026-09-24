"""DCR-CVA: end-to-end center corrector, gradient-isolated anchored ranker.

Only the corrector's CDF controls physical center selection. Rank supervision
cannot flow into image/CVA features, CDF, or the immutable proposal generator.
The entire corrector remains trainable by its own CDF (optionally relative) loss.
"""
from __future__ import annotations
import torch
from torch import nn
from models.economicgrasp_cva_centers import CenterHypothesisCVA
from dcr_cva_common import RankResidualHead


class DecoupledCenterRankingCVA(nn.Module):
    def __init__(self, reference, offsets_mm, group_chunk=512, rank_hidden=128,
                 rank_bound=.5, rank_seed=2032):
        super().__init__()
        self.corrector = CenterHypothesisCVA(reference,offsets_mm,group_chunk)
        dim = int(reference.kview_config.head_model_dim)
        # Do not change corrector/dropout RNG by adding an auxiliary head.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(rank_seed)
            self.ranker = RankResidualHead(dim,rank_hidden,rank_bound)

    @property
    def zero(self):
        return self.corrector.zero

    @property
    def reference(self):
        return self.corrector.reference

    def forward(self, batch, bundle=None, **kwargs):
        logits,bundle,depth,latent = self.corrector(batch,bundle,return_features=True,**kwargs)
        native_score = bundle['actions'][self.zero,:,0]
        residual = self.ranker(latent,logits,native_score,self.corrector.offsets_mm,self.zero)
        return logits,residual,bundle,depth

    def learned_state(self):
        return {'corrector':self.corrector.learned_state(),
                'ranker':{k:v.detach().cpu() for k,v in self.ranker.state_dict().items()}}

    def load_learned_state(self,state):
        if set(state) != {'corrector','ranker'}:
            raise RuntimeError('DCR learned-state contract mismatch')
        self.corrector.load_learned_state(state['corrector'])
        self.ranker.load_state_dict(state['ranker'],strict=True)

    def warm_start_corrector(self,e1_state):
        self.corrector.load_learned_state(e1_state)
