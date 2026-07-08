# Hugging Face Hub integration (#31)

**Status:** approved-by-implementer (autonomous batch)
**Issue:** [#31](https://github.com/minigraphx/godot-native-rl/issues/31) (Backlog item 50)
**Date:** 2026-07-08

## Problem

There is no one-command way to **share** a trained ncnn deploy model or **fetch** a pretrained one.
The deploy artifacts are just local files (`*.ncnn.param` / `*.ncnn.bin` + optional sidecars), so
distribution is manual copy-paste. Backlog item 50 asks for a Python CLI wrapping `huggingface_hub`
to push/pull models to the Hub — mirroring godot_rl's Hub story and strengthening the "deploy
everywhere" distribution track.

## Design

A single CLI, `scripts/hf_hub.py`, with two subcommands, following the `tune_optuna.py` precedent:
stdlib-only at module load (so the pure helpers unit-test with no dep), `huggingface_hub` imported
**lazily** inside the push/pull functions, and isolated in an opt-in `requirements-hub.txt`
(installed on top of any venv: `.venv-train/bin/pip install -r requirements-hub.txt`).

### Commands

- `push <path> <repo_id> [--token T] [--private] [--message M] [--no-card]`
  `<path>` is a model directory **or** a stem (e.g. `examples/chase_the_target/models/chase_the_target`).
  Collects the ncnn artifacts (+ sidecars), generates a `README.md` model card (unless `--no-card`),
  and `HfApi.upload_folder`s a temp staging dir to `repo_id` (created if missing, `repo_type=model`).
- `pull <repo_id> <dest_dir> [--token T] [--revision R]`
  `snapshot_download`s the repo into `dest_dir` (filtered to model artifacts).

Token resolution: `--token` → `HF_TOKEN` env → the cached `huggingface_hub` login (its default).

### Pure helpers (unit-tested, no `huggingface_hub`)

- `collect_model_files(path) -> list[Path]` — given a dir or a stem, return the deploy files to
  upload. A **param+bin pair is required** (raises `FileNotFoundError` otherwise). Includes known
  sidecars when present: `*.recurrent.json`, `*_action_dist.json`, `*_vecnorm*.json` /
  `*_vecnormalize*.json`, `*.ncnn.param`/`*.ncnn.bin`, and any `*_golden.json` is **excluded** (test
  fixture, not a deploy artifact). Stem mode selects only files sharing that stem; dir mode takes all
  matching files in the dir.
- `stem_of(path)` — the model stem (strips `.ncnn.param`/`.ncnn.bin`/known suffixes).
- `build_model_card(repo_id, files, *, base_model=None) -> str` — a Markdown model card with YAML
  front-matter (`library_name: godot-native-rl`, `tags: [godot, ncnn, reinforcement-learning,
  deep-reinforcement-learning, godot-native-rl]`, `license: mit`) + a body listing the shipped files
  and the two-line "deploy in Godot via NcnnRunner" usage. Pure string builder.
- `parse_args(argv)` — argparse; validated, returns the namespace.

### Thin hub wrappers (lazy import)

- `push_to_hub(path, repo_id, token, private, message, write_card) -> str` — collect files, stage
  them (+ card) in a `tempfile.TemporaryDirectory`, `create_repo(exist_ok=True)` + `upload_folder`,
  return the repo URL.
- `pull_from_hub(repo_id, dest, token, revision) -> Path` — `snapshot_download` with an
  `allow_patterns` filter for model artifacts; return the local dir.

These call `_hf_api(token)` which lazily imports `huggingface_hub` and constructs `HfApi`. Tests
**monkeypatch `_hf_api`** (and the lazy `create_repo`/`upload_folder`/`snapshot_download` seams) to
assert the orchestration calls the right APIs with the right args — no network, no token needed.

## Testing

- `test/python/test_hf_hub.py` (stdlib unittest, auto-discovered by `run_tests.sh`):
  - `collect_model_files`: dir mode, stem mode, sidecar inclusion, golden exclusion, missing-pair error.
  - `build_model_card`: front-matter tags present, files listed, deterministic.
  - `parse_args`: push/pull parsing, defaults, required-arg errors.
  - push/pull orchestration with a **fake HfApi** injected — asserts `create_repo` + `upload_folder`
    / `snapshot_download` called with expected `repo_id`/`repo_type`/staged files (no network).
- Live push/pull is manual (no HF token/network in CI) — documented in the script header, same
  honest treatment as the xvfb-only paths.

## Files

| File | Change |
|---|---|
| `scripts/hf_hub.py` | **new** CLI (`push`/`pull` + pure helpers) |
| `requirements-hub.txt` | **new** isolated `huggingface_hub` add-on |
| `test/python/test_hf_hub.py` | **new** pure-helper + mocked-orchestration tests |
| `CLAUDE.md`, `docs/BACKLOG.md` | document + check off item 50 |

Closes #31.
