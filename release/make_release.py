"""Build the public release: `src/lfo`, its tests, and the scripts, READMEs and results behind the three papers.

The release is a path-filtered clone of this repository. History is kept for the released paths (original dates),
so each pre-registration commit still precedes its results; hashes change, and `HASHES.md` maps the development
hashes quoted in the READMEs to the new ones. Not released: `docs/` (plans), `STATUS.md`, `CLAUDE.md`, the
foundation reproduction results, and `experiments/C/theory/` apart from the modules the released scripts import.

    pip install git-filter-repo
    python release/make_release.py OUT_DIR [--test]      # OUT_DIR must not exist

The release commit uses the README in `release/README.md`. Nothing is pushed.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("src/lfo", "tests", "experiments/A/layer", "experiments/A/deep", "experiments/A/process",
           "experiments/A/paper", "experiments/C/e1_audit", "experiments/C/e5_probe", "experiments/C/paper",
           "experiments/D")
FILES = ("LICENSE", "pyproject.toml", ".gitignore", "experiments/make_index.py",
         "experiments/foundation/README.md", "experiments/C/theory/H1_H2.md")


def git(*args, cwd=ROOT, env=None):
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


def tracked():
    return [Path(p) for p in git("ls-files").split("\n") if p]


def release_paths() -> list[str]:
    """The released folders and files, plus every experiment module their scripts import (transitively)."""
    files = tracked()
    keep = {f for f in files if any(str(f).startswith(d + "/") for d in FOLDERS)} | {Path(f) for f in FILES}
    modules = {}
    for f in files:
        if f.suffix == ".py" and str(f).startswith("experiments/"):
            modules.setdefault(f.stem, []).append(f)
    todo = [f for f in keep if f.suffix == ".py"]
    while todo:
        tree = ast.parse((ROOT / todo.pop()).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                for g in modules.get(name.split(".")[0], []):
                    if g not in keep:
                        keep.add(g)
                        todo.append(g)
    missing = [str(p) for p in keep if not (ROOT / p).exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    return sorted(str(p) for p in keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--test", action="store_true", help="run pytest in the release")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists():
        sys.exit(f"{out} exists")
    if git("status", "--porcelain").strip():
        sys.exit("working tree not clean: commit first (the release is built from HEAD)")
    paths = release_paths()
    git("clone", "--no-local", "--quiet", str(ROOT), str(out))
    plist = out.parent / (out.name + ".paths")
    plist.write_text("\n".join(paths) + "\n")
    subprocess.run([sys.executable, "-m", "git_filter_repo", "--paths-from-file", str(plist), "--force", "--quiet"],
                   cwd=out, check=True)
    plist.unlink()

    # development hash -> release hash, for the commits that survive the filter
    rows = []
    for line in (out / ".git" / "filter-repo" / "commit-map").read_text().split("\n")[1:]:
        if not line.strip():
            continue
        old, new = line.split()
        if set(new) == {"0"}:
            continue
        date, subject = git("log", "-1", "--format=%ad%x09%s", "--date=short", old).strip().split("\t", 1)
        rows.append((date, old[:7], new[:7], subject.replace("|", "/")))
    rows.sort()
    (out / "HASHES.md").write_text("\n".join(
        ["# Development hashes", "",
         "The READMEs quote commit hashes of the development repository. This table maps each commit that touched a",
         "released file to its commit here (same date and message).", "",
         "| date | development | release | message |", "|---|---|---|---|"]
        + [f"| {d} | `{o}` | `{n}` | {s} |" for d, o, n, s in rows]) + "\n")
    shutil.copy(ROOT / "release" / "README.md", out / "README.md")
    subprocess.run([sys.executable, "experiments/make_index.py"], cwd=out, check=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="Kee-Youn Yoo", GIT_COMMITTER_NAME="Kee-Youn Yoo")
    git("add", "-A", cwd=out)
    git("commit", "-q", "-m", "Release 0.0.1: README for the release, hash map, index with release hashes", cwd=out, env=env)
    n_py = sum(p.endswith(".py") for p in paths)
    print(f"release at {out}: {len(paths)} files ({n_py} .py), {len(rows)} commits kept")
    if args.test:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=out, check=True)


if __name__ == "__main__":
    main()
