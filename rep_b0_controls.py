"""Retrained visual-attribution controls, not inference-time branch deletion.

full: exact B0.
no_image: exact same network and geometric inputs, but zero image CONTENT.
independent_rgb: same per-grasp image reader/action/offset/scorer; the same
Transformer is applied to length-one sequences so candidates cannot interact.
All parameters/initial shared weights are preserved across the three controls.
"""
from __future__ import annotations

import torch
from rep_b_model import RepBModel

CONTROLS = ('full', 'no_image', 'independent_rgb')


class RepB0Control(RepBModel):
    def __init__(self, control='full', **kwargs):
        if control not in CONTROLS:
            raise ValueError(f'Unknown attribution control: {control}')
        if kwargs.pop('variant', 'B0') != 'B0':
            raise ValueError('Attribution controls require B0, not B1/B2')
        super().__init__(variant='B0', **kwargs)
        self.control = control

    def encode(self, data, depth=None, return_components=False):
        if self.control == 'full':
            return super().encode(data, depth, return_components)
        if self.control == 'no_image':
            # Keep the gripper geometry, visibility mask, pixel projection and
            # ALL learned layers. Remove only visual content, during train/test.
            no_content = dict(data, image_feature=torch.zeros_like(data['image_feature']))
            return super().encode(no_content, depth, return_components)

        valid = data['valid'].bool()
        a = self._safe_actions(data['actions'].float(), valid, int(data['zero_index']))
        k, q = a.shape[:2]
        hw = data['depth'].shape[-2:]
        image = self.image_reader(data['image_feature'].float()[None], a.reshape(-1, 17),
                                  data['K'].float()[None], hw).reshape(k, q, self.dim)
        action = self.action_embed(a[..., 1:16])
        offset = self._offset_features(data['offsets_mm'], q)
        tokens = image + action + offset
        # Identical architecture/parameters, but each attention sequence has
        # one candidate. Invalid candidates cannot contaminate another ray.
        rep = self.relational(tokens.reshape(k*q, 1, self.dim)).reshape(k, q, self.dim)
        rep = torch.where(valid[..., None], rep, torch.zeros_like(rep))
        components = dict(image=image, action=action, offset=offset, prior=None,
                          pre_relation=tokens)
        return (rep, components) if return_components else rep
