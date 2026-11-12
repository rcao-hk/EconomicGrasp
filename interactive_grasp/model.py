"""2x2 ablation: independent/update, read again, interaction, interact then read.

All variants are permutation equivariant over grasp candidates. Attention uses
padding masks only (no geometric radius, graph adjacency or leader prior).
The stage-1 score is independent sigmoid quality, not a set-normalized policy.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import nn

VARIANTS = {"A": (False, False), "B": (False, True),
            "C": (True, False), "D": (True, True)}


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int
    memory_dim: int
    variant: str = "D"
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 2
    read_chunk: int = 256
    seed: int = 0

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError("variant must be A/B/C/D")
        if min(self.feature_dim, self.memory_dim, self.hidden_dim, self.heads,
               self.layers, self.read_chunk) < 1 or self.hidden_dim % self.heads:
            raise ValueError("Invalid model dimensions")


class PerActionRead(nn.Module):
    """A different memory for every action; never attends other actions' probes."""
    def __init__(self, width: int, memory_dim: int, heads: int, chunk: int):
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(memory_dim, width)
        self.v = nn.Linear(memory_dim, width)
        self.out = nn.Linear(width, width)
        self.heads, self.chunk = heads, chunk

    def forward(self, query, memory, memory_valid, action_valid):
        B, N, C = query.shape
        P = memory.shape[-2]
        flat_q = query.reshape(B*N, C)
        flat_m = memory.reshape(B*N, P, -1)
        valid = memory_valid.reshape(B*N, P).bool()
        action_valid = action_valid.reshape(-1)
        values = []
        for start in range(0, B*N, self.chunk):
            end = min(start+self.chunk, B*N)
            mask = valid[start:end] & action_valid[start:end, None]
            any_valid = mask.any(-1)
            # Avoid NaNs at all-invalid actions, then explicitly suppress output
            # (including output projection bias). Invalid memory never leaks.
            safe = mask.clone()
            safe[~any_valid, 0] = True
            mem = torch.where(mask[..., None], flat_m[start:end], 0.0)
            q = self.q(flat_q[start:end]).reshape(-1, self.heads, 1, C//self.heads)
            k = self.k(mem).reshape(-1, P, self.heads, C//self.heads).transpose(1, 2)
            v = self.v(mem).reshape(-1, P, self.heads, C//self.heads).transpose(1, 2)
            # Explicit attention, no stochastic dropout; mask semantics unambiguous.
            logits = torch.matmul(q, k.transpose(-1, -2)) / (C//self.heads)**0.5
            logits = logits.masked_fill(~safe[:, None, None, :], -torch.inf)
            attended = torch.matmul(logits.softmax(-1), v).reshape(-1, C)
            delta = self.out(attended)
            values.append(torch.where(any_valid[:, None], delta, 0.0))
        return torch.cat(values).reshape(B, N, C)


class UpdateBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, block_seed: int):
        super().__init__()
        interaction, read = VARIANTS[cfg.variant]
        self.interaction, self.read = interaction, read
        C = cfg.hidden_dim
        self.mix_norm = nn.LayerNorm(C)
        self.read_norm = nn.LayerNorm(C)
        self.ffn_norm = nn.LayerNorm(C)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(block_seed)
            if interaction:
                self.mix = nn.MultiheadAttention(C, cfg.heads, batch_first=True, dropout=0.0)
            else:
                # ~same active parameter budget as Q/K/V/O, no dead attention weights.
                self.mix = nn.Sequential(nn.Linear(C, 2*C), nn.GELU(), nn.Linear(2*C, C))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(block_seed + 1)
            if read:
                self.reader = PerActionRead(C, cfg.memory_dim, cfg.heads, cfg.read_chunk)
            else:
                # Same weight count as Q/K/V/O read; biases differ slightly.
                self.reader = nn.Sequential(nn.Linear(C, C+cfg.memory_dim), nn.GELU(),
                                            nn.Linear(C+cfg.memory_dim, C))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(block_seed + 2)
            self.ffn = nn.Sequential(nn.Linear(C, 2*C), nn.GELU(), nn.Linear(2*C, C))

    def forward(self, x, geometry, memory, memory_valid, action_valid,
                bypass_interaction_query=False):
        old = x
        n = self.mix_norm(x)
        if self.interaction:
            safe_actions = action_valid.clone()
            all_pad = ~safe_actions.any(-1)
            safe_actions[all_pad, 0] = True
            # Geometry affects query/key matching, not an explicit pairwise bias.
            qk = n + geometry
            delta = self.mix(qk, qk, n, key_padding_mask=~safe_actions,
                             need_weights=False)[0]
            delta = torch.where(action_valid[..., None], delta, 0.0)
        else:
            delta = self.mix(n)
        x = x + delta
        if self.read:
            # Optional *evaluation-only* mechanism intervention: preserve the
            # interaction residual but query memory using the pre-interaction state.
            read_q = old if bypass_interaction_query else x
            update = self.reader(self.read_norm(read_q), memory, memory_valid, action_valid)
        else:
            update = self.reader(self.read_norm(x))
        x = x + update
        x = x + self.ffn(self.ffn_norm(x))
        return torch.where(action_valid[..., None], x, 0.0)


class InteractiveGraspRefiner(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Component-local RNG: equal common initialization independent of which
        # A/B/C/D branches consume random numbers. No side effects on data RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            self.feature = nn.Sequential(nn.LayerNorm(cfg.feature_dim),
                                         nn.Linear(cfg.feature_dim, cfg.hidden_dim))
            self.geometry = nn.Sequential(nn.Linear(15, cfg.hidden_dim), nn.GELU(),
                                          nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        blocks = []
        for i in range(cfg.layers):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed + 1009*(i+1))
                blocks.append(UpdateBlock(cfg, cfg.seed + 1009*(i+1)))
        self.blocks = nn.ModuleList(blocks)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed + 65537)
            self.final_norm = nn.LayerNorm(cfg.hidden_dim)
            self.head = nn.Linear(cfg.hidden_dim, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def action_encoding(grasps):
        # Complete physical action only: width,height,insertion,R(9),camera t(3).
        # Scores and object ids are NOT tokens. Unit normalization is not a
        # neighborhood/competition threshold; translation remains in camera metres.
        geom = grasps[..., 1:16].clone()
        geom[..., :3] = geom[..., :3] / 0.1
        return geom

    def forward(self, features, grasps, base_logit, memory=None, memory_valid=None,
                action_valid=None, bypass_interaction_query=False):
        if features.ndim != 3 or grasps.shape != (*features.shape[:2], 17):
            raise ValueError("Expected features [B,N,F] and grasps [B,N,17]")
        B, N, F = features.shape
        if N < 1 or F != self.cfg.feature_dim or base_logit.shape != (B, N):
            raise ValueError("Invalid feature/logit dimensions")
        if action_valid is None:
            action_valid = torch.ones((B, N), dtype=torch.bool, device=features.device)
        if action_valid.shape != (B, N) or action_valid.dtype != torch.bool:
            raise ValueError("action_valid must be bool [B,N]")
        if bypass_interaction_query and self.training:
            raise ValueError("Pre-interaction-query intervention is inference-only")
        if VARIANTS[self.cfg.variant][1]:
            if memory is None or memory_valid is None or memory.ndim != 4:
                raise ValueError("Read variants require [B,N,P,E] memory and mask")
            if (memory.shape[:2] != (B, N) or memory.shape[-1] != self.cfg.memory_dim
                    or memory.shape[2] < 1 or memory_valid.shape != memory.shape[:3]
                    or memory_valid.dtype != torch.bool):
                raise ValueError("Bad evidence memory contract")
        # Mask before normalization, including padded NaN values from callers.
        features = torch.where(action_valid[..., None], features, 0.0)
        grasps = torch.where(action_valid[..., None], grasps, 0.0)
        geo = self.geometry(self.action_encoding(grasps))
        x = self.feature(features) + geo
        x = torch.where(action_valid[..., None], x, 0.0)
        for block in self.blocks:
            x = block(x, geo, memory, memory_valid, action_valid, bypass_interaction_query)
        delta = self.head(self.final_norm(x)).squeeze(-1)
        delta = torch.where(action_valid, delta, 0.0)
        return base_logit + delta

    def describe(self):
        return {**asdict(self.cfg), "parameters": sum(p.numel() for p in self.parameters()),
                "changes_physical_action": False, "score_feedback": False,
                "graph_or_nms_mask": False, "candidate_index_embedding": False}
