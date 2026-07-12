"""Unit tests for the pure helpers + injected-client orchestration in scripts/hf_hub.py (#31).

hf_hub is stdlib-only at module load (`huggingface_hub` is imported lazily inside the push/pull
seams), so these tests run with no `huggingface_hub` installed: they cover model-file collection
(dir + stem modes, sidecar inclusion, golden exclusion, missing-pair errors), the model-card
builder, arg parsing, and the push/pull orchestration via an injected fake client (no network)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hf_hub  # noqa: E402  (stdlib-only at import; no dep probe needed)


def _touch(d: Path, name: str, body: str = "x") -> Path:
    p = d / name
    p.write_text(body)
    return p


class TestCollectModelFiles(unittest.TestCase):
    def test_dir_mode_gathers_deploy_files_excludes_golden(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _touch(d, "a.ncnn.param")
            _touch(d, "a.ncnn.bin")
            _touch(d, "a_action_dist.json")
            _touch(d, "a_golden.json")   # test fixture — must NOT be uploaded
            _touch(d, "README.md")       # not a deploy artifact
            names = sorted(p.name for p in hf_hub.collect_model_files(d))
            self.assertEqual(names, ["a.ncnn.bin", "a.ncnn.param", "a_action_dist.json"])

    def test_dir_mode_multiple_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for stem in ("a", "b"):
                _touch(d, f"{stem}.ncnn.param")
                _touch(d, f"{stem}.ncnn.bin")
            names = sorted(p.name for p in hf_hub.collect_model_files(d))
            self.assertEqual(names, ["a.ncnn.bin", "a.ncnn.param", "b.ncnn.bin", "b.ncnn.param"])

    def test_stem_mode_selects_only_that_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _touch(d, "chase.ncnn.param")
            _touch(d, "chase.ncnn.bin")
            _touch(d, "chase.recurrent.json")
            _touch(d, "chase_es.ncnn.param")   # a DIFFERENT model sharing a prefix
            _touch(d, "chase_es.ncnn.bin")
            names = sorted(p.name for p in hf_hub.collect_model_files(d / "chase"))
            self.assertEqual(names, ["chase.ncnn.bin", "chase.ncnn.param", "chase.recurrent.json"])

    def test_stem_mode_accepts_a_param_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _touch(d, "m.ncnn.param")
            _touch(d, "m.ncnn.bin")
            names = sorted(p.name for p in hf_hub.collect_model_files(d / "m.ncnn.param"))
            self.assertEqual(names, ["m.ncnn.bin", "m.ncnn.param"])

    def test_vecnorm_sidecar_rides_along(self):
        # #363: the VecNormalize obs-norm stats JSON follows the stem-coupled `_vecnorm.json`
        # convention, so it uploads/downloads with the net it belongs to.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _touch(d, "rover.ncnn.param")
            _touch(d, "rover.ncnn.bin")
            _touch(d, "rover_vecnorm.json")
            names = sorted(p.name for p in hf_hub.collect_model_files(d / "rover"))
            self.assertEqual(names, ["rover.ncnn.bin", "rover.ncnn.param", "rover_vecnorm.json"])

    def test_pull_patterns_include_vecnorm(self):
        self.assertIn("*_vecnorm.json", hf_hub.PULL_PATTERNS)

    def test_missing_bin_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _touch(d, "a.ncnn.param")
            with self.assertRaises(FileNotFoundError):
                hf_hub.collect_model_files(d)

    def test_missing_param_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _touch(d, "a.ncnn.bin")
            with self.assertRaises(FileNotFoundError):
                hf_hub.collect_model_files(d)


class TestStemOf(unittest.TestCase):
    def test_strips_ncnn_suffixes(self):
        self.assertEqual(hf_hub.stem_of(Path("models/chase.ncnn.param")), "chase")
        self.assertEqual(hf_hub.stem_of(Path("models/chase.ncnn.bin")), "chase")
        self.assertEqual(hf_hub.stem_of(Path("models/chase")), "chase")
        self.assertEqual(hf_hub.stem_of(Path("models/rover_vecnorm.json")), "rover")


class TestBuildModelCard(unittest.TestCase):
    def test_card_has_frontmatter_tags_and_lists_files(self):
        files = [Path("m.ncnn.param"), Path("m.ncnn.bin")]
        card = hf_hub.build_model_card("org/chase-agent", files)
        self.assertTrue(card.startswith("---"), "card starts with YAML front-matter")
        self.assertIn("library_name: godot-native-rl", card)
        self.assertIn("ncnn", card)
        self.assertIn("reinforcement-learning", card)
        self.assertIn("org/chase-agent", card)
        self.assertIn("m.ncnn.param", card)
        self.assertIn("m.ncnn.bin", card)

    def test_deterministic(self):
        files = [Path("m.ncnn.param"), Path("m.ncnn.bin")]
        self.assertEqual(hf_hub.build_model_card("o/r", files), hf_hub.build_model_card("o/r", files))

    def test_vecnorm_deploy_hint(self):
        # #363: a policy shared WITH obs-norm stats gets a card hint naming the controller
        # property, so the puller wires it (deploying without the stats is silently wrong).
        files = [Path("m.ncnn.param"), Path("m.ncnn.bin"), Path("m_vecnorm.json")]
        card = hf_hub.build_model_card("o/r", files)
        self.assertIn("obs_norm_stats_path", card)
        self.assertIn("m_vecnorm.json", card)
        self.assertNotIn("obs_norm_stats_path", hf_hub.build_model_card("o/r", files[:2]))


class TestParseArgs(unittest.TestCase):
    def test_push_defaults(self):
        ns = hf_hub.parse_args(["push", "models/chase", "org/chase"])
        self.assertEqual(ns.command, "push")
        self.assertEqual(ns.path, "models/chase")
        self.assertEqual(ns.repo_id, "org/chase")
        self.assertFalse(ns.private)
        self.assertTrue(ns.card)          # card on by default
        self.assertIsNone(ns.token)

    def test_push_flags(self):
        ns = hf_hub.parse_args(["push", "d", "o/r", "--private", "--no-card", "--message", "hi"])
        self.assertTrue(ns.private)
        self.assertFalse(ns.card)
        self.assertEqual(ns.message, "hi")

    def test_pull_defaults(self):
        ns = hf_hub.parse_args(["pull", "org/chase", "out"])
        self.assertEqual(ns.command, "pull")
        self.assertEqual(ns.repo_id, "org/chase")
        self.assertEqual(ns.dest, "out")
        self.assertIsNone(ns.revision)


class _FakeApi:
    """Injected stand-in for huggingface_hub.HfApi — records calls; captures the staged folder's
    contents at upload time (before push_to_hub's TemporaryDirectory is cleaned up)."""

    def __init__(self):
        self.created = None
        self.uploaded = None
        self.staged = None

    def create_repo(self, repo_id, repo_type="model", private=False, exist_ok=False, token=None):
        self.created = {"repo_id": repo_id, "repo_type": repo_type, "private": private,
                        "exist_ok": exist_ok}

    def upload_folder(self, repo_id, folder_path, repo_type="model", commit_message=None, token=None):
        self.uploaded = {"repo_id": repo_id, "repo_type": repo_type, "commit_message": commit_message}
        self.staged = sorted(os.listdir(folder_path))


class TestPushOrchestration(unittest.TestCase):
    def _model_dir(self, tmp):
        d = Path(tmp)
        _touch(d, "m.ncnn.param")
        _touch(d, "m.ncnn.bin")
        return d

    def test_push_stages_files_and_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._model_dir(tmp)
            api = _FakeApi()
            url = hf_hub.push_to_hub(d, "org/chase", api=api)
            self.assertEqual(api.created["repo_id"], "org/chase")
            self.assertEqual(api.created["repo_type"], "model")
            self.assertTrue(api.created["exist_ok"])
            self.assertEqual(api.uploaded["repo_id"], "org/chase")
            self.assertIn("m.ncnn.param", api.staged)
            self.assertIn("m.ncnn.bin", api.staged)
            self.assertIn("README.md", api.staged)   # model card staged by default
            self.assertEqual(url, "https://huggingface.co/org/chase")

    def test_push_no_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._model_dir(tmp)
            api = _FakeApi()
            hf_hub.push_to_hub(d, "org/chase", api=api, write_card=False)
            self.assertNotIn("README.md", api.staged)

    def test_push_private_and_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._model_dir(tmp)
            api = _FakeApi()
            hf_hub.push_to_hub(d, "org/chase", api=api, private=True, message="v1")
            self.assertTrue(api.created["private"])
            self.assertEqual(api.uploaded["commit_message"], "v1")


class TestPullOrchestration(unittest.TestCase):
    def test_pull_filters_and_returns_dest(self):
        calls = {}

        def fake_download(repo_id, local_dir, revision=None, allow_patterns=None, token=None,
                          repo_type="model"):
            calls.update(repo_id=repo_id, local_dir=local_dir, revision=revision,
                         allow_patterns=allow_patterns, repo_type=repo_type)
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            return local_dir

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            got = hf_hub.pull_from_hub("org/chase", dest, download=fake_download)
            self.assertEqual(Path(got), dest)
            self.assertEqual(calls["repo_id"], "org/chase")
            self.assertEqual(Path(calls["local_dir"]), dest)
            self.assertEqual(calls["repo_type"], "model")
            self.assertIn("*.ncnn.param", calls["allow_patterns"])
            self.assertIn("*.ncnn.bin", calls["allow_patterns"])


if __name__ == "__main__":
    unittest.main()
