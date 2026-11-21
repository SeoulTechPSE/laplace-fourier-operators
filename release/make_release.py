"""Build the public release: `src/lfo`, its tests, and the scripts, READMEs and results behind the three papers.

The release is a path-filtered clone of this repository. History is kept for the released paths (original dates),
so each pre-registration commit still precedes its results; hashes change, and `HASHES.md` maps the development
hashes quoted in the READMEs to the new ones. Not released: `docs/` (plans), `STATUS.md`, `CLAUDE.md`, the
foundation reproduction results, `experiments/C/e1_audit/E1_report.md` (internal report with the follow-up plan),
and `experiments/C/theory/` (a follow-up study, unpublished).

The follow-up study stays out of the release: the `C/theory` modules that released scripts import are added in the
release commit only (no history, docstring replaced by a neutral line), and commit messages that mention the
follow-up (E2–E4, H1–H3, propositions, CRB, certificates, EIV) are replaced by a neutral message.

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
FILES = ("LICENSE", "pyproject.toml", ".gitignore", "experiments/make_index.py", "experiments/foundation/README.md")
EXCLUDE = ("experiments/C/e1_audit/E1_report.md",)
LATE = "experiments/C/theory/"     # imported modules from here enter in the release commit only
STUB = ('"""Helper module of a follow-up identification study (not part of this release); included because released '
        'scripts import it."""')
MESSAGE_CALLBACK = r"""
import re
if re.search(rb"\b(E[234]|H[1-3]|[Pp]roposition|CRB|Cram|certificate|EIV|[Ii]nput noise|output noise)\b", message):
    scope = message.split(b":", 1)[0] if b":" in message.split(b"\n", 1)[0] else b"C"
    message = scope + b": update (message withheld: it concerns an unpublished follow-up study)\n"
return message
"""


def git(*args, cwd=ROOT, env=None):
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout


def tracked():
    return [Path(p) for p in git("ls-files").split("\n") if p]


def release_paths() -> list[str]:
    """The released folders and files, plus every experiment module their scripts import (transitively)."""
    files = tracked()
    keep = {f for f in files if any(str(f).startswith(d + "/") for d in FOLDERS)} | {Path(f) for f in FILES}
    keep -= {Path(f) for f in EXCLUDE}
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


def stub_docstring(src: str) -> str:
    doc = ast.parse(src).body[0]
    lines = src.split("\n")
    return "\n".join(lines[:doc.lineno - 1] + [STUB] + lines[doc.end_lineno:])


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
    late = [p for p in paths if p.startswith(LATE)]
    paths = [p for p in paths if not p.startswith(LATE)]
    git("clone", "--no-local", "--quiet", str(ROOT), str(out))
    plist = out.parent / (out.name + ".paths")
    plist.write_text("\n".join(paths) + "\n")
    subprocess.run([sys.executable, "-m", "git_filter_repo", "--paths-from-file", str(plist), "--force", "--quiet",
                    "--message-callback", MESSAGE_CALLBACK],
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
        date, subject = git("log", "-1", "--format=%ad%x09%s", "--date=short", new, cwd=out).strip().split("\t", 1)
        rows.append((date, old[:7], new[:7], subject.replace("|", "/")))
    rows.sort()
    (out / "HASHES.md").write_text("\n".join(
        ["# Development hashes", "",
         "The READMEs quote commit hashes of the development repository. This table maps each commit that touched a",
         "released file to its commit here (same date and message).", "",
         "| date | development | release | message |", "|---|---|---|---|"]
        + [f"| {d} | `{o}` | `{n}` | {s} |" for d, o, n, s in rows]) + "\n")
    shutil.copy(ROOT / "release" / "README.md", out / "README.md")
    for p in late:
        (out / p).parent.mkdir(parents=True, exist_ok=True)
        (out / p).write_text(stub_docstring(git("show", f"HEAD:{p}")))
    subprocess.run([sys.executable, "experiments/make_index.py"], cwd=out, check=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="Kee-Youn Yoo", GIT_COMMITTER_NAME="Kee-Youn Yoo")
    git("add", "-A", cwd=out)
    git("commit", "-q", "-m", "Release 0.0.1: README for the release, hash map, index with release hashes", cwd=out, env=env)
    n_py = sum(p.endswith(".py") for p in paths)
    print(f"release at {out}: {len(paths)} files with history ({n_py} .py), {len(late)} late modules, "
          f"{len(rows)} commits kept")
    if args.test:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=out, check=True)


if __name__ == "__main__":
    main()
