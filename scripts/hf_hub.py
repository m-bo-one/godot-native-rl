#!/usr/bin/env python3
"""Push/pull trained ncnn deploy models to the Hugging Face Hub (#31, backlog item 50).

One command to share a trained ncnn model — the deploy artifacts (`*.ncnn.param` / `*.ncnn.bin` +
the stem-coupled sidecars `*.recurrent.json` / `*_action_dist.json` / `*_vecnorm.json`, #363) plus
an auto-generated model card — or to fetch a pretrained one back:

    .venv-train/bin/python scripts/hf_hub.py push examples/chase_the_target/models/chase_the_target org/chase-agent
    .venv-train/bin/python scripts/hf_hub.py pull org/chase-agent examples/chase_the_target/models

`<path>` for push is a model DIRECTORY (uploads every model in it) or a STEM / param path (uploads
just that one model + its sidecars). Token resolution: `--token` -> `HF_TOKEN` env -> the cached
`huggingface_hub` login.

Dependency isolation (mirrors tune_optuna.py, #113): `huggingface_hub` is NOT in requirements-train.txt.
Install it on demand: `.venv-train/bin/pip install -r requirements-hub.txt`. It is imported LAZILY
inside the push/pull seams, so the pure helpers below (and their tests) run with no dep installed.

Live push/pull needs a Hub token + network, so it is exercised manually, not in CI; the pure helpers
and the orchestration (via an injected client) are unit-tested in test/python/test_hf_hub.py.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

# Stem-coupled deploy artifacts. A model is identified by its `.ncnn.param`/`.ncnn.bin` PAIR; the
# JSON sidecars ride alongside when present. `*_golden.json` (a test fixture) is deliberately NOT
# collected. `_vecnorm.json` (#363) is the stem-coupled VecNormalize obs-norm stats convention:
# `export_vecnormalize.py --stem <model-stem>` writes it, so a VecNormalize-trained policy ships
# WITH the stats it needs (deploying without them produces wrong actions silently).
DEPLOY_SUFFIXES = (".ncnn.param", ".ncnn.bin", ".recurrent.json", "_action_dist.json", "_vecnorm.json")

# What `pull` fetches from a model repo (artifacts + the card), skipping anything else in the repo.
PULL_PATTERNS = ["*.ncnn.param", "*.ncnn.bin", "*.recurrent.json", "*_action_dist.json",
                 "*_vecnorm.json", "*.md"]

CARD_TAGS = (
    "godot",
    "ncnn",
    "reinforcement-learning",
    "deep-reinforcement-learning",
    "godot-native-rl",
)


# --- Pure helpers (unit-tested; no huggingface_hub needed) ---

def stem_of(path) -> str:
    """The model stem: the basename with a known deploy suffix (or a single plain suffix) stripped."""
    name = Path(path).name
    for suf in DEPLOY_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem if Path(name).suffix else name


def collect_model_files(path) -> list[Path]:
    """Return the deploy files to upload for `path`.

    Directory mode (`path` is a dir): every file whose name ends with a DEPLOY_SUFFIX (all models in
    the dir). Stem mode (anything else — a stem like `.../chase` or a `.../chase.ncnn.param`): only the
    files sharing that exact stem. Raises FileNotFoundError unless a `.ncnn.param` + `.ncnn.bin` pair
    is found (a model needs both to deploy).
    """
    path = Path(path)
    if path.is_dir():
        found = sorted(p for p in path.iterdir()
                       if p.is_file() and p.name.endswith(DEPLOY_SUFFIXES))
    else:
        stem = stem_of(path)
        parent = path.parent
        found = sorted(parent / (stem + suf) for suf in DEPLOY_SUFFIXES
                       if (parent / (stem + suf)).is_file())
    names = [p.name for p in found]
    if not any(n.endswith(".ncnn.param") for n in names):
        raise FileNotFoundError("no .ncnn.param found for %s" % path)
    if not any(n.endswith(".ncnn.bin") for n in names):
        raise FileNotFoundError("no .ncnn.bin found for %s" % path)
    return found


def build_model_card(repo_id: str, files, *, base_model: str | None = None) -> str:
    """Build the README.md model card (YAML front-matter + body) for a set of deploy files. Pure."""
    tags = "\n".join("  - %s" % t for t in CARD_TAGS)
    front = ["---", "library_name: godot-native-rl", "license: mit", "tags:", tags]
    if base_model:
        front.append("base_model: %s" % base_model)
    front.append("---")
    file_list = "\n".join("- `%s`" % n for n in sorted(p.name for p in files))
    vecnorm_hint = ""
    vecnorms = sorted(p.name for p in files if p.name.endswith("_vecnorm.json"))
    if vecnorms:
        vecnorm_hint = (
            "\n## Observation normalization (required)\n\n"
            "This policy was trained with SB3 `VecNormalize` — set the controller's\n"
            "`obs_norm_stats_path` to the bundled stats so the obs mean/std is replayed game-side\n"
            "before inference (without it the policy produces wrong actions silently):\n\n"
            + "\n".join("    obs_norm_stats_path = \"res://<your_models_dir>/%s\"" % n for n in vecnorms)
            + "\n")
    body = f"""
# {repo_id}

A trained reinforcement-learning policy for **native on-device deployment in Godot** via
[godot-native-rl](https://github.com/minigraphx/godot-native-rl) — statically-linked ncnn inference,
no Python or managed runtime at deploy (web, mobile, console, desktop, edge).

## Files

{file_list}

## Deploy in Godot

Copy the `.ncnn.param` / `.ncnn.bin` (and any sidecars) into your project and point an
`NcnnAIController2D` / `NcnnAIController3D` (or `NcnnRunner`) at them — see the
[deploy guide](https://github.com/minigraphx/godot-native-rl/blob/main/docs/dev/DEVELOPMENT.md).
{vecnorm_hint}

    .venv-train/bin/python scripts/hf_hub.py pull {repo_id} <local_models_dir>
"""
    return "\n".join(front) + "\n" + body


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hf_hub", allow_abbrev=False,
                                description="Push/pull trained ncnn models to the Hugging Face Hub.")
    sub = p.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="upload a model dir or stem to a Hub repo")
    push.add_argument("path", help="model directory, or a stem / .ncnn.param path")
    push.add_argument("repo_id", help="target Hub repo, e.g. org/chase-agent")
    push.add_argument("--token", default=None, help="Hub token (else HF_TOKEN env / cached login)")
    push.add_argument("--private", action="store_true", help="create the repo private")
    push.add_argument("--message", default=None, help="commit message")
    push.add_argument("--no-card", dest="card", action="store_false",
                      help="do not generate/upload a README.md model card")
    push.set_defaults(card=True)

    pull = sub.add_parser("pull", help="download a model repo into a local dir")
    pull.add_argument("repo_id", help="source Hub repo, e.g. org/chase-agent")
    pull.add_argument("dest", help="local directory to download into")
    pull.add_argument("--token", default=None, help="Hub token (else HF_TOKEN env / cached login)")
    pull.add_argument("--revision", default=None, help="branch / tag / commit to fetch")
    return p.parse_args(argv)


# --- Hub wrappers (huggingface_hub imported lazily; injectable seams for tests) ---

def _hf_api(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download
    return snapshot_download(**kwargs)


def push_to_hub(path, repo_id: str, *, token: str | None = None, private: bool = False,
                message: str | None = None, write_card: bool = True, api=None) -> str:
    """Collect a model's deploy files, stage them (+ a model card) and upload to `repo_id`.

    `api` is an injected huggingface_hub-HfApi-like client (create_repo + upload_folder); defaults to
    a real `HfApi`. Returns the repo URL.
    """
    files = collect_model_files(path)
    api = api or _hf_api(token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    with tempfile.TemporaryDirectory() as staging:
        stage = Path(staging)
        for f in files:
            shutil.copy2(f, stage / f.name)
        if write_card:
            (stage / "README.md").write_text(build_model_card(repo_id, files))
        api.upload_folder(repo_id=repo_id, folder_path=str(stage), repo_type="model",
                          commit_message=message or "Upload ncnn model (godot-native-rl)", token=token)
    return "https://huggingface.co/%s" % repo_id


def pull_from_hub(repo_id: str, dest, *, token: str | None = None, revision: str | None = None,
                  download=None) -> Path:
    """Download the model artifacts of `repo_id` into `dest`. `download` is an injectable
    snapshot_download-like seam (defaults to the real one). Returns the local directory."""
    download = download or _snapshot_download
    local = download(repo_id=repo_id, local_dir=str(dest), revision=revision,
                     allow_patterns=PULL_PATTERNS, token=token, repo_type="model")
    return Path(local)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "push":
        url = push_to_hub(args.path, args.repo_id, token=args.token, private=args.private,
                          message=args.message, write_card=args.card)
        print("Pushed to:", url)
    elif args.command == "pull":
        local = pull_from_hub(args.repo_id, args.dest, token=args.token, revision=args.revision)
        print("Pulled to:", local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
