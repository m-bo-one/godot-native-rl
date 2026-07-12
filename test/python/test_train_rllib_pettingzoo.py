"""Unit tests for the pure helpers of scripts/train_rllib_pettingzoo.py (#123).

The module must import without ray/gymnasium/godot_rl (lazy heavy imports); the space-squeeze
helpers operate on gymnasium space objects, so those cases are dep-probe-guarded (#141/#148 —
only third-party deps in the try, so a missing dep skips while a real ImportError stays loud).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_rllib_pettingzoo as tr  # noqa: E402

try:
    import numpy as np  # noqa: F401, E402
    from gymnasium import spaces  # noqa: E402
    HAVE_GYM = True
except ImportError:
    HAVE_GYM = False

needs_gym = unittest.skipUnless(HAVE_GYM, "numpy/gymnasium not installed")


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = tr.parse_args([])
        self.assertEqual(cfg.base_port, 11008)
        self.assertEqual(cfg.policies, ("seeker", "hider"))
        self.assertEqual(cfg.experiment, "hide_seek_rllib")

    def test_overrides(self):
        cfg = tr.parse_args(["--timesteps", "5000", "--policies", "a", "b", "c",
                             "--experiment", "x", "--train_dir", "/tmp/t"])
        self.assertEqual(cfg.timesteps, 5000)
        self.assertEqual(cfg.policies, ("a", "b", "c"))


class TestPolicyMapping(unittest.TestCase):
    def test_mapping_from_names(self):
        self.assertEqual(tr.mapping_from_names(["seeker", "hider", "seeker", "hider"]),
                         {0: "seeker", 1: "hider", 2: "seeker", 3: "hider"})

    def test_empty_names_raises(self):
        with self.assertRaises(ValueError):
            tr.mapping_from_names([])

    def test_validate_policies_subset_ok(self):
        tr.validate_policies(["seeker", "hider"], ("seeker", "hider"))  # no raise

    def test_validate_policies_unknown_wire_name_raises(self):
        # A scene emitting a policy name the CLI didn't declare must fail loud — otherwise
        # RLlib would train a module the mapping can't route.
        with self.assertRaises(ValueError) as ctx:
            tr.validate_policies(["seeker", "ghost"], ("seeker", "hider"))
        self.assertIn("ghost", str(ctx.exception))


class TestSpaceSqueeze(unittest.TestCase):
    @needs_gym
    def test_obs_dict_to_box(self):
        box = spaces.Box(low=-1.0, high=1.0, shape=(15,))
        self.assertIs(tr.squeeze_obs_space(spaces.Dict({"obs": box})), box)

    @needs_gym
    def test_obs_without_obs_key_raises(self):
        with self.assertRaises(ValueError):
            tr.squeeze_obs_space(spaces.Dict({"camera_2d": spaces.Box(low=0, high=255, shape=(2, 2, 3))}))

    @needs_gym
    def test_action_tuple_single_discrete(self):
        d = spaces.Discrete(5)
        self.assertIs(tr.squeeze_action_space(spaces.Tuple((d,))), d)

    @needs_gym
    def test_action_tuple_multi_key_raises(self):
        with self.assertRaises(ValueError):
            tr.squeeze_action_space(spaces.Tuple((spaces.Discrete(5), spaces.Discrete(2))))

    @needs_gym
    def test_action_non_discrete_raises(self):
        with self.assertRaises(ValueError):
            tr.squeeze_action_space(spaces.Tuple((spaces.Box(low=-1, high=1, shape=(2,)),)))


class TestAgentPolicyRegistryFile(unittest.TestCase):
    # The env -> policy_mapping_fn channel must survive cloudpickle-by-value (ray's register_env
    # copies __main__ globals into each pickled function, so a module-level registry the env fills
    # is INVISIBLE to the mapping fn — found live). A file is the source of truth.
    def test_round_trip_preserves_int_agent_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_policies.json"
            tr.write_agent_policies(path, {0: "seeker", 1: "hider"})
            self.assertEqual(tr.read_agent_policies(path), {0: "seeker", 1: "hider"})

    def test_make_policy_mapping_fn_reads_and_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_policies.json"
            tr.write_agent_policies(path, {0: "seeker", 1: "hider"})
            fn = tr.make_policy_mapping_fn(path)
            self.assertEqual(fn(0), "seeker")
            self.assertEqual(fn(1, "episode-arg-ignored"), "hider")
            with self.assertRaises(RuntimeError):
                fn(7)

    def test_mapping_fn_missing_file_fails_loud(self):
        fn = tr.make_policy_mapping_fn("/nonexistent/agent_policies.json")
        with self.assertRaises(RuntimeError):
            fn(0)


class TestEnvMeta(unittest.TestCase):
    def test_round_trip(self):
        # The .sh reads this JSON to drive the per-policy export without hardcoding dims.
        meta = tr.build_env_meta(obs_dim=15, nvec=[5], policies=("seeker", "hider"))
        self.assertEqual(meta, {"obs_dim": 15, "nvec": [5], "policies": ["seeker", "hider"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env_meta.json"
            tr.write_env_meta(path, meta)
            self.assertEqual(json.loads(path.read_text()), meta)


if __name__ == "__main__":
    unittest.main()
