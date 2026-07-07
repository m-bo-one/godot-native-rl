#!/usr/bin/env python3
"""SB3 BaseFeaturesExtractor wrapping the shared AttentionEncoder (#46/#259 M3).

With ``net_arch=[]`` the SB3 policy's ``action_net`` becomes a single ``Linear(embed_dim, n_act)``,
so the deploy actor is exactly this extractor's encoder followed by that head — the M2 export
contract (scripts/export_statedict_to_ncnn.attention_policy_layers). godot_rl's
StableBaselinesGodotEnv gives a Dict obs ``{"obs": Box}`` (hence MultiInputPolicy), so ``forward``
unwraps the ``"obs"`` key; a plain Box space is supported for tests.
"""
from __future__ import annotations

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from attention_encoder import AttentionEncoder


def _obs_width(observation_space) -> int:
    if isinstance(observation_space, spaces.Dict):
        return int(observation_space.spaces["obs"].shape[-1])
    return int(observation_space.shape[-1])


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim: int = 16, num_heads: int = 2,
                 n_entities: int = 6, feat: int = 4):
        super().__init__(observation_space, features_dim=embed_dim)
        width = _obs_width(observation_space)
        assert n_entities * feat + n_entities == width, (
            "obs width %d != n*f+n (n=%d f=%d)" % (width, n_entities, feat))
        self._is_dict = isinstance(observation_space, spaces.Dict)
        self.encoder = AttentionEncoder(n_entities, feat, embed_dim, num_heads)

    def forward(self, observations):
        flat = observations["obs"] if self._is_dict else observations
        return self.encoder(flat)
