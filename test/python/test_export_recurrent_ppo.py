"""Unit tests for the RecurrentPPO actor-extraction seam (#378).

Guarded on the opt-in sb3-contrib add-on (requirements-recurrent.txt) like the jax/tune/hub
tests. Covers the export contract WITHOUT pnnx/ncnn/Godot: the extracted (obs, h, c) wrapper
must reproduce model.predict(deterministic=True) step-for-step across a carried sequence
(episode-start reset = zero state), and the stacked-LSTM guard must fail loud. The
ONNX->pnnx->ncnn half is covered by the trainer pipeline + the committed-net regression.
"""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

try:
    import numpy as np
    import torch
    import gymnasium as gym
    from sb3_contrib import RecurrentPPO
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(HAVE_DEPS, "sb3-contrib not installed (requirements-recurrent.txt)")

OBS, N_ACTIONS, HIDDEN = 5, 4, 16


if HAVE_DEPS:
    class _TinyEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
        action_space = gym.spaces.Discrete(N_ACTIONS)

        def reset(self, *, seed=None, options=None):
            return np.zeros(OBS, np.float32), {}

        def step(self, action):
            return np.zeros(OBS, np.float32), 0.0, False, False, {}

    def _make_model(n_lstm_layers=1):
        return RecurrentPPO(
            "MlpLstmPolicy", _TinyEnv(), seed=7, n_steps=8, batch_size=8,
            policy_kwargs=dict(lstm_hidden_size=HIDDEN, n_lstm_layers=n_lstm_layers,
                               net_arch=dict(pi=[8], vf=[8])),
        )


@needs_deps
class TestRecurrentActorExtraction(unittest.TestCase):
    def test_wrapper_matches_predict_over_carried_sequence(self):
        from export_recurrent_ppo import build_actor
        model = _make_model()
        actor = build_actor(model.policy.eval()).eval()
        rng = np.random.default_rng(3)
        obs_seq = [rng.standard_normal(OBS).astype(np.float32) for _ in range(6)]

        h = torch.zeros(1, 1, HIDDEN)
        c = torch.zeros(1, 1, HIDDEN)
        wrapper_actions = []
        with torch.no_grad():
            for o in obs_seq:
                logits, h, c = actor(torch.from_numpy(o).reshape(1, 1, OBS), h, c)
                wrapper_actions.append(int(np.argmax(logits.reshape(-1).numpy())))

        state = None
        episode_start = np.ones((1,), bool)
        sb3_actions = []
        for o in obs_seq:
            a, state = model.predict(o, state=state, episode_start=episode_start,
                                     deterministic=True)
            episode_start = np.zeros((1,), bool)
            sb3_actions.append(int(a))
        self.assertEqual(wrapper_actions, sb3_actions)

    def test_state_actually_carries(self):
        # The same obs twice with carried vs zeroed state must differ once state is nonzero —
        # guards against a wrapper that silently drops the recurrence.
        from export_recurrent_ppo import build_actor
        model = _make_model()
        actor = build_actor(model.policy.eval()).eval()
        rng = np.random.default_rng(5)
        o = torch.from_numpy(rng.standard_normal(OBS).astype(np.float32)).reshape(1, 1, OBS)
        with torch.no_grad():
            _, h, c = actor(o * 3.0, torch.zeros(1, 1, HIDDEN), torch.zeros(1, 1, HIDDEN))
            carried, _, _ = actor(o, h, c)
            zeroed, _, _ = actor(o, torch.zeros(1, 1, HIDDEN), torch.zeros(1, 1, HIDDEN))
        self.assertFalse(torch.allclose(carried, zeroed),
                         "carried state had no effect — recurrence dropped in the wrapper")

    def test_stacked_lstm_fails_loud(self):
        import tempfile
        from export_recurrent_ppo import export_recurrent_actor
        model = _make_model(n_lstm_layers=2)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "stacked.zip"
            model.save(ckpt)
            with self.assertRaises(SystemExit):
                export_recurrent_actor(str(ckpt), Path(tmp), "stacked")


if __name__ == "__main__":
    unittest.main()
