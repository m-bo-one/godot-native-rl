#!/usr/bin/env python3
"""Self-play pool bookkeeping for the alternating-phase orchestrator (#29).

    selfplay_phase.py register-snapshot --pool-dir models/selfplay_pool/hider --name hider_gen1

Appends a member to the pool's ELO ledger (pool.json) at the current learner rating (league
convention: a frozen copy starts where the learner left off), creating the ledger if absent.
Stdlib only; the pure helper is unit-tested in test/python/test_selfplay_phase.py.
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile

DEFAULT_RATING = 1200.0


def register_snapshot(ledger: dict, name: str, rating=None) -> dict:
    """Pure: return a new ledger with `name` added at `rating` (default: learner_rating)."""
    members = dict(ledger.get("members", {}))
    if name in members:
        raise ValueError(f"snapshot '{name}' already registered")
    learner = float(ledger.get("learner_rating", DEFAULT_RATING))
    members[name] = {"rating": float(rating) if rating is not None else learner, "games": 0}
    return {"members": members, "learner_rating": learner}


def unique_member_name(existing, base: str) -> str:
    """Pure: `base` if free, else the first available `base_2`, `base_3`, ... (#371).

    The #189 mid-run freezer names snapshots by update index, which restarts from the same schedule
    on every run. Reruns against a persistent pool_dir would otherwise reuse a prior run's name —
    overwriting its weights and then crashing on the duplicate registration. Uniquifying keeps the
    old snapshot intact and grows the pool additively."""
    existing = set(existing)
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def read_member_names(pool_dir) -> set:
    """Existing pool member names, or an empty set if the ledger is absent/unreadable (#371)."""
    ledger_path = pathlib.Path(pool_dir) / "pool.json"
    if not ledger_path.exists():
        return set()
    try:
        return set(json.loads(ledger_path.read_text()).get("members", {}).keys())
    except (json.JSONDecodeError, OSError, ValueError):
        return set()


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write `text` to `path` via a temp file + os.replace in the same dir (#371), mirroring
    checkpoints.write_manifest — so a SelfPlayManager reader consuming the pool WHILE it grows
    (the whole point of #189) can never observe torn JSON."""
    fd, tmp = tempfile.mkstemp(prefix=".pool-", suffix=".json", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o644)  # mkstemp is 0600; pool.json is a normal human-readable file
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def register_snapshot_file(pool_dir, name: str, rating=None) -> dict:
    """File-level registration (create-ledger-if-absent + register + write): THE one writer of
    pool.json, shared by this CLI and the #189 mid-run freezer so the on-disk ledger contract has
    a single definition. The write is atomic (#371). Returns the updated ledger."""
    pool_dir = pathlib.Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = pool_dir / "pool.json"
    ledger = {"members": {}, "learner_rating": DEFAULT_RATING}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
    ledger = register_snapshot(ledger, name, rating)
    _atomic_write_text(ledger_path, json.dumps(ledger, indent=2))
    return ledger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("register-snapshot")
    reg.add_argument("--pool-dir", required=True)
    reg.add_argument("--name", required=True)
    reg.add_argument("--rating", type=float, default=None)
    args = parser.parse_args(argv)

    ledger = register_snapshot_file(args.pool_dir, args.name, args.rating)
    print(f"registered '{args.name}' at rating {ledger['members'][args.name]['rating']} "
          f"in {pathlib.Path(args.pool_dir) / 'pool.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
