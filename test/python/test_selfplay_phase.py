import json
import pathlib
import tempfile
import unittest

from scripts.selfplay_phase import (
    DEFAULT_RATING,
    read_member_names,
    register_snapshot,
    register_snapshot_file,
    unique_member_name,
)


class TestRegisterSnapshot(unittest.TestCase):
    def test_new_member_enters_at_learner_rating(self):
        ledger = {"members": {}, "learner_rating": 1337.0}
        out = register_snapshot(ledger, "gen1")
        self.assertEqual(out["members"]["gen1"], {"rating": 1337.0, "games": 0})
        self.assertEqual(out["learner_rating"], 1337.0)

    def test_explicit_rating_honored(self):
        out = register_snapshot({"members": {}}, "gen1", rating=900.0)
        self.assertEqual(out["members"]["gen1"]["rating"], 900.0)

    def test_default_rating_when_ledger_empty(self):
        out = register_snapshot({}, "gen1")
        self.assertEqual(out["members"]["gen1"]["rating"], DEFAULT_RATING)

    def test_existing_member_rejected(self):
        ledger = {"members": {"gen1": {"rating": 1200.0, "games": 3}}}
        with self.assertRaises(ValueError):
            register_snapshot(ledger, "gen1")

    def test_does_not_mutate_input(self):
        ledger = {"members": {}, "learner_rating": 1500.0}
        register_snapshot(ledger, "gen1")
        self.assertEqual(ledger["members"], {})


class TestUniqueMemberName(unittest.TestCase):
    def test_no_collision_returns_base(self):
        self.assertEqual(unique_member_name(set(), "seeker_live_u000002"), "seeker_live_u000002")

    def test_single_collision_appends_2(self):
        self.assertEqual(unique_member_name({"gen1"}, "gen1"), "gen1_2")

    def test_chained_collisions_find_first_free(self):
        self.assertEqual(unique_member_name({"gen1", "gen1_2", "gen1_3"}, "gen1"), "gen1_4")

    def test_accepts_any_iterable(self):
        self.assertEqual(unique_member_name(["gen1"], "gen1"), "gen1_2")


class TestReadMemberNames(unittest.TestCase):
    def test_absent_ledger_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(read_member_names(d), set())

    def test_reads_registered_names(self):
        with tempfile.TemporaryDirectory() as d:
            register_snapshot_file(d, "gen1")
            register_snapshot_file(d, "gen2")
            self.assertEqual(read_member_names(d), {"gen1", "gen2"})

    def test_corrupt_ledger_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            (pathlib.Path(d) / "pool.json").write_text("{ this is not json")
            self.assertEqual(read_member_names(d), set())

    def test_wrong_shape_ledger_is_empty(self):
        # Valid JSON but the wrong shape (members not a dict, or a non-dict top level) must still
        # yield an empty set, not crash freeze_snapshot mid-run (#371 robustness).
        for content in ['{"members": []}', "[1, 2, 3]", '"just a string"', "42"]:
            with tempfile.TemporaryDirectory() as d:
                (pathlib.Path(d) / "pool.json").write_text(content)
                self.assertEqual(read_member_names(d), set(), content)


class TestRegisterSnapshotFile(unittest.TestCase):
    def test_create_if_absent_then_append_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            l1 = register_snapshot_file(d, "gen1")
            self.assertEqual(set(l1["members"]), {"gen1"})
            l2 = register_snapshot_file(d, "gen2")
            self.assertEqual(set(l2["members"]), {"gen1", "gen2"})
            on_disk = json.loads((pathlib.Path(d) / "pool.json").read_text())
            self.assertEqual(set(on_disk["members"]), {"gen1", "gen2"})

    def test_atomic_write_leaves_no_temp_files(self):
        # The write goes through a temp file + os.replace (mirrors checkpoints.write_manifest),
        # so a concurrent SelfPlayManager reader can never see torn JSON. The observable regression
        # guard: no stray temp file is left behind, and the ledger is valid JSON.
        with tempfile.TemporaryDirectory() as d:
            register_snapshot_file(d, "gen1")
            leftovers = sorted(p.name for p in pathlib.Path(d).iterdir() if p.name != "pool.json")
            self.assertEqual(leftovers, [], f"atomic write left temp files: {leftovers}")

    def test_duplicate_name_still_raises(self):
        # register_snapshot_file keeps protecting the ledger; the trainer avoids the crash by
        # uniquifying the name BEFORE calling it (see TestRerunCollision).
        with tempfile.TemporaryDirectory() as d:
            register_snapshot_file(d, "gen1")
            with self.assertRaises(ValueError):
                register_snapshot_file(d, "gen1")


class TestRerunCollision(unittest.TestCase):
    """#371: a second run restarts the update schedule from the same indices, so the base snapshot
    name collides with a prior run's pool member. Uniquifying before export keeps the old
    snapshot's weights + rating intact and grows the pool additively (instead of overwriting the
    weights then crashing the checkpoint-less trainer on the duplicate registration)."""

    def test_rerun_uniquifies_and_preserves_prior_member(self):
        with tempfile.TemporaryDirectory() as d:
            # run 1 froze this snapshot; it accrued ELO history.
            register_snapshot_file(d, "seeker_live_u000002", rating=1300.0)
            # run 2's freeze of the same update index: uniquify against the on-disk ledger.
            base = "seeker_live_u000002"
            name = unique_member_name(read_member_names(d), base)
            self.assertEqual(name, "seeker_live_u000002_2")
            register_snapshot_file(d, name)
            on_disk = json.loads((pathlib.Path(d) / "pool.json").read_text())
            self.assertEqual(
                set(on_disk["members"]),
                {"seeker_live_u000002", "seeker_live_u000002_2"},
            )
            # run 1's snapshot rating is untouched — the ledger entry still matches its weights.
            self.assertEqual(on_disk["members"]["seeker_live_u000002"]["rating"], 1300.0)


if __name__ == "__main__":
    unittest.main()
