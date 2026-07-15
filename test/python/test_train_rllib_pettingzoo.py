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


class TestPolicyIdentity(unittest.TestCase):
    # Policy identity travels in the PettingZoo agent id ("<policy>_<index>") — the mapping fn
    # is a pure string parse, immune to the cloudpickle-by-value trap (see docs/dev/gotchas.md).
    def test_agent_ids_from_names(self):
        self.assertEqual(tr.agent_ids_from_names(["seeker", "hider", "seeker", "hider"]),
                         ["seeker_0", "hider_1", "seeker_2", "hider_3"])

    def test_empty_names_raises(self):
        with self.assertRaises(ValueError):
            tr.agent_ids_from_names([])

    def test_policy_of_parses_and_ignores_episode_arg(self):
        self.assertEqual(tr.policy_of("seeker_0"), "seeker")
        self.assertEqual(tr.policy_of("hider_13", "episode-arg-ignored"), "hider")
        # Policy names containing underscores survive (split on the LAST underscore).
        self.assertEqual(tr.policy_of("team_red_2"), "team_red")

    def test_policy_of_malformed_id_fails_loud(self):
        with self.assertRaises(RuntimeError):
            tr.policy_of("no-index")
        with self.assertRaises(RuntimeError):
            tr.policy_of(0)

    def test_inner_index_round_trip(self):
        ids = tr.agent_ids_from_names(["seeker", "hider"])
        self.assertEqual([tr.inner_index_of(a) for a in ids], [0, 1])

    def test_validate_policies_subset_ok(self):
        tr.validate_policies(["seeker", "hider"], ("seeker", "hider"))  # no raise

    def test_validate_policies_unknown_wire_name_raises(self):
        # A scene emitting a policy name the CLI didn't declare must fail loud — otherwise
        # RLlib would train a module the mapping can't route.
        with self.assertRaises(ValueError) as ctx:
            tr.validate_policies(["seeker", "ghost"], ("seeker", "hider"))
        self.assertIn("ghost", str(ctx.exception))

    def test_validate_policies_warns_on_unused_declared(self):
        # #374: a declared policy that never appears on the wire is a config error — RLlib creates
        # an untrained module and the export loop would ship it from init weights. Warn (not raise).
        import io
        from contextlib import redirect_stderr
        buf = io.StringIO()
        with redirect_stderr(buf):
            tr.validate_policies(["seeker", "seeker"], ("seeker", "hider"))  # hider never mapped
        self.assertIn("hider", buf.getvalue())

    def test_export_policies_are_wire_observed_only(self):
        # #374: meta['policies'] drives the export loop, so it must list only policies that actually
        # got batches (appeared on the wire) — never a declared-but-unmapped one.
        self.assertEqual(tr.export_policies(["seeker", "hider", "seeker"]), ["hider", "seeker"])
        self.assertEqual(tr.export_policies(["seeker", "seeker"]), ["seeker"])


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
