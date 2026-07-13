"""Unit tests for the truncation-aware godot_rl bridge (#12).

The pure splitter runs bare; the env-subclass case is dep-probe-guarded on godot_rl/numpy
(#141/#148) and exercises step_recv over an injected message (no socket)."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import godot_env_truncation as gt  # noqa: E402

try:
    import numpy  # noqa: F401
    import godot_rl  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

needs_deps = unittest.skipUnless(HAVE_DEPS, "numpy/godot_rl not installed")


class TestSplitDoneTruncated(unittest.TestCase):
    def test_wire_split(self):
        # done stays = terminated OR truncated on the wire; the split recovers Gymnasium flags.
        term, trunc = gt.split_done_truncated([True, True, False], [True, False, False])
        self.assertEqual(term, [False, True, False])
        self.assertEqual(trunc, [True, False, False])

    def test_missing_wire_field_treats_ends_as_terminal(self):
        # Pre-#12 scenes / foreign servers don't send `truncated`: all ends count as terminal
        # (truncated stays False) — strictly better than upstream's done-duplication, and it
        # matches GodotParallelEnv's previous behavior.
        term, trunc = gt.split_done_truncated([True, False], None)
        self.assertEqual(term, [True, False])
        self.assertEqual(trunc, [False, False])


@needs_deps
class TestTruncationAwareGodotEnv(unittest.TestCase):
    def _env_with_response(self, response):
        env = gt.TruncationAwareGodotEnv.__new__(gt.TruncationAwareGodotEnv)  # no socket
        env._get_json_dict = lambda: dict(response)
        env._process_obs = lambda o: o
        return env

    def test_step_recv_splits_flags(self):
        env = self._env_with_response({
            "obs": [{"obs": [0.0]}], "reward": [1.0],
            "done": [True], "truncated": [True], "info": [{}],
        })
        _obs, _r, term, trunc, _info = env.step_recv()
        self.assertEqual(term, [False])
        self.assertEqual(trunc, [True])

    def test_step_recv_without_truncated_field(self):
        env = self._env_with_response({
            "obs": [{"obs": [0.0]}], "reward": [0.5], "done": [True],
        })
        _obs, _r, term, trunc, info = env.step_recv()
        self.assertEqual(term, [True])
        self.assertEqual(trunc, [False])
        self.assertEqual(info, [{}])  # upstream's default_info fallback preserved


if __name__ == "__main__":
    unittest.main()
