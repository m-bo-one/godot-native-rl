import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_rllib_to_torchscript as ex  # noqa: E402


class TestActorLogitLayout(unittest.TestCase):
    def test_single_discrete(self):
        self.assertEqual(ex.actor_logit_layout([5]), (5, [5]))

    def test_multi_discrete(self):
        self.assertEqual(ex.actor_logit_layout([3, 2, 4]), (9, [3, 2, 4]))

    def test_empty_nvec_raises(self):
        with self.assertRaises(ValueError):
            ex.actor_logit_layout([])

    def test_non_positive_entry_raises(self):
        with self.assertRaises(ValueError):
            ex.actor_logit_layout([5, 0])


class TestPickModuleId(unittest.TestCase):
    # #123: multi-policy checkpoints hold one module per policy; --module_id selects it.
    def test_explicit_id(self):
        self.assertEqual(ex.pick_module_id(["seeker", "hider"], "hider"), "hider")

    def test_explicit_id_missing_raises_and_lists_ids(self):
        with self.assertRaises(RuntimeError) as ctx:
            ex.pick_module_id(["seeker", "hider"], "ghost")
        self.assertIn("seeker", str(ctx.exception))

    def test_default_policy_preferred(self):
        self.assertEqual(ex.pick_module_id(["default_policy"], None), "default_policy")

    def test_single_key_fallback(self):
        self.assertEqual(ex.pick_module_id(["only"], None), "only")

    def test_ambiguous_names_module_id_flag(self):
        with self.assertRaises(RuntimeError) as ctx:
            ex.pick_module_id(["seeker", "hider"], None)
        self.assertIn("--module_id", str(ctx.exception))


class TestLatestCheckpoint(unittest.TestCase):
    def test_returns_newest_by_mtime(self):
        # train_rllib.py saves checkpoints as <train_dir>/<experiment>/checkpoint_NNNNNN/.
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = os.path.join(tmp, "chase_rllib")
            older = os.path.join(exp_dir, "checkpoint_000001")
            newer = os.path.join(exp_dir, "checkpoint_000002")
            os.makedirs(older)
            os.makedirs(newer)
            past = time.time() - 100
            os.utime(older, (past, past))
            self.assertEqual(ex.latest_checkpoint(tmp, "chase_rllib"), newer)

    def test_no_checkpoints_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "chase_rllib"))
            with self.assertRaises(FileNotFoundError):
                ex.latest_checkpoint(tmp, "chase_rllib")

    def test_missing_experiment_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                ex.latest_checkpoint(tmp, "chase_rllib")


if __name__ == "__main__":
    unittest.main()
