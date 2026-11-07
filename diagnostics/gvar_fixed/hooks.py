"""Temporary inference-only GVAR interventions. No parameters or source patches."""
from __future__ import annotations

from contextlib import contextmanager
import torch


def residual_permutation(queries: torch.Tensor, output: torch.Tensor, permutation):
    """Swap only reader updates, never depth embeddings, heads or action labels."""
    if queries.ndim != 5 or output.shape != queries.shape:
        raise ValueError('Expected [B,Q,A,D,H] reader queries and output')
    d = queries.shape[3]
    p = tuple(int(v) for v in permutation)
    if sorted(p) != list(range(d)):
        raise ValueError('Depth permutation must be a bijection over all depth bins')
    if p == tuple(range(d)):
        return output  # identity must preserve every bit; no subtract/add roundoff
    delta = output - queries
    indices = torch.tensor(p, device=output.device, dtype=torch.long)
    return queries + delta.index_select(3, indices)


@contextmanager
def reader_intervention(reader, condition: str):
    """native, rgb_fixed, perm_identity, perm_cycle, or geometry_bias_<mm>.

    Native RGB, metadata, mask and geometry remain dynamic except where named.
    geometry_bias changes only GVAR relative geometry, not upstream RayRoPE,
    action location, RGB sampling positions or executed width.
    """
    if reader is None:
        if condition != 'native':
            raise ValueError('Point-only model has no GVAR reader to intervene on')
        yield
        return
    if reader.training or torch.is_grad_enabled():
        raise RuntimeError('Reader diagnostics require eval() and inference/no_grad mode')
    old = getattr(reader, 'counterfactual_mode', None)
    if old is None or not hasattr(reader, 'set_counterfactual'):
        raise TypeError('Reader does not implement the audited counterfactual contract')
    handles = []
    try:
        reader.set_counterfactual('r1' if condition == 'rgb_fixed' else 'r0')
        if condition in ('perm_identity', 'perm_cycle'):
            def swap(_module, args, output):
                queries = args[0]
                d = queries.shape[3]
                p = list(range(d)) if condition == 'perm_identity' else list(range(1, d)) + [0]
                return residual_permutation(queries, output, p)
            handles.append(reader.register_forward_hook(swap))
        elif condition.startswith('geometry_bias_'):
            bias = float(condition.removeprefix('geometry_bias_')) * .001
            if not torch.isfinite(torch.tensor(bias)):
                raise ValueError('Nonfinite geometry bias')
            def perturb(_module, args):
                if len(args) != 2 or not isinstance(args[1], dict):
                    raise RuntimeError('Reader context call contract changed')
                context = dict(args[1]); depth = context['depth']
                context['depth'] = torch.where(torch.isfinite(depth) & (depth > 0),
                                               (depth + bias).clamp_min(1e-4), depth)
                return args[0], context
            handles.append(reader.register_forward_pre_hook(perturb))
        elif condition not in ('native', 'rgb_fixed'):
            raise ValueError(f'Unknown intervention: {condition}')
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        reader.set_counterfactual(old)
