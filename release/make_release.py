"""Build the public release: `src/lfo`, its tests, and the scripts, READMEs and results behind the three papers.

The release is a path-filtered clone of this repository. History is kept for the released paths (original dates),
so each pre-registration commit still precedes its results; hashes change, and `HASHES.md` maps the development
hashes quoted in the READMEs to the new ones. Not released: `docs/` (plans), `STATUS.md`, `CLAUDE.md`, the
foundation reproduction results, `experiments/C/e1_audit/E1_report.md` (internal report with the follow-up plan),
and `experiments/C/theory/` (a follow-up study, unpublished).

The follow-up study stays out of the release: of the `C/theory` modules that released scripts import, only the
top-level code those scripts use (transitively) is kept (`shake_late`), added in the release commit only (no history,
no comments or inner docstrings, module docstring replaced by a neutral line); file references to the
follow-up are replaced by "a follow-up study" in every version (REPLACE_TEXT), as is the C paper's summary of the
follow-up's results (one neutral sentence; the sentences that relied on it are reworded), and commit messages that mention the
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
FILES = ("LICENSE", "CITATION.cff", ".github/workflows/tests.yml", "pyproject.toml", ".gitignore", "experiments/make_index.py", "experiments/foundation/README.md")
EXCLUDE = ("experiments/C/e1_audit/E1_report.md",)
LATE = "experiments/C/theory/"     # imported modules from here enter in the release commit only
STUB = ('"""Helper module of a follow-up identification study (not part of this release); included because released '
        'scripts import it."""')
# cross-references to the follow-up in released files, replaced in every historical version
REPLACE_TEXT = """experiments/C/theory/README.md (E2 paragraph)==>a follow-up study
A follow-up study (to appear) shows that the E1 data do identify these poles:==>A follow-up study (to appear) addresses whether the E1 data identify these poles.
regex:\\n- A training recipe without prior knowledge of the poles recovers[^\\n]*==>
regex:\\n- A truth-free certificate built on a first-order-hold refit[^\\n]*==>
regex:\\n- Under relative output noise the limit is the Cramér–Rao bound[^\\n]*==>
regex:\\nThe fast real poles are therefore missed by the published recipes, not hidden by the data\\.\\n==>
regex:\\n- The follow-up recipe was tested on the tied single layer only\\.==>
### Fast real poles are not found, although the data identify them {#sec-fastreal}==>### Fast real poles are not found {#sec-fastreal}
3. Fast real poles are not found by any published recipe, although the data identify them.==>3. Fast real poles are not found by any published recipe; their profiles are flat within the layer's own floor.
`experiments/C/theory/H1_H2.md`==>a follow-up study
`experiments/C/theory/README.md`==>a follow-up study
experiments/C/theory/README.md==>a follow-up study
`H1_H2.md`==>a follow-up study
Proposition 8's==>a follow-up result's
Proposition 8==>a follow-up result
Proposition 7==>a follow-up result
"""
MESSAGE_CALLBACK = r"""
import re
first = message.split(b"\n", 1)[0]
terms = rb"E[234]|H[1-3]|CRB|Cram|certificate|EIV|[Ii]nput noise|output noise"
if not first.startswith(b"A:"):        # (A)'s own propositions (domain consistency) are not the follow-up's
    terms += rb"|[Pp]roposition"
if re.search(rb"\b(" + terms + rb")\b", message):
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


def neutral_module(src: str) -> str:
    """Replace the module docstring by STUB; drop function / class docstrings and comments (they record the follow-up)."""
    import io
    import tokenize
    toks = [t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type != tokenize.COMMENT]
    src = tokenize.untokenize(toks)
    src = "\n".join(line.rstrip() for line in src.split("\n"))
    lines = src.split("\n")
    tree = ast.parse(src)
    cut = {}                     # first line -> (last line, replacement)
    doc = tree.body[0]
    cut[doc.lineno] = (doc.end_lineno, STUB)
    for node in ast.walk(tree):  # function and class docstrings go too
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            d = node.body[0]
            indent = lines[d.lineno - 1][:d.col_offset]
            cut[d.lineno] = (d.end_lineno, indent + "pass" if len(node.body) == 1 else None)
    out, i = [], 1
    while i <= len(lines):
        if i in cut:
            end, rep = cut[i]
            if rep is not None:
                out.append(rep)
            i = end + 1
        else:
            out.append(lines[i - 1])
            i += 1
    return "\n".join(out)


def _bound(node):
    """Names a top-level statement binds."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return set()


def _late_aliases(tree, stems):
    """alias -> late module stem, for `import stem [as alias]`."""
    return {(a.asname or a.name): a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
            for a in n.names if a.name in stems}


def shake_late(paths, late):
    """Keep, in each late module, only the top-level code that released scripts use (transitively).

    Roots are the names released scripts import from a late module (`from m import x`, `import m as M` with `M.x`).
    Within a late module a kept statement pulls in every top-level statement binding a name it references, and
    `M.x` on another late module adds x to that module's roots. `sys.path` set-up, `from __future__` and the module
    docstring are kept; everything else (the follow-up's own code, its CLI and report writers) is dropped. Returns
    {path: source} for the modules with any kept code."""
    stems = {Path(p).stem: p for p in late}
    trees = {p: ast.parse(git("show", f"HEAD:{p}")) for p in late}
    roots = {p: set() for p in late}

    def add_refs(tree_nodes, aliases):
        changed = False
        for node in tree_nodes:
            for n in ast.walk(node):
                if isinstance(n, ast.ImportFrom) and n.module in stems:
                    new = {a.name for a in n.names} - roots[stems[n.module]]
                    if new:
                        roots[stems[n.module]] |= new
                        changed = True
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                    q = stems[aliases[n.value.id]]
                    if n.attr not in roots[q]:
                        roots[q].add(n.attr)
                        changed = True
        return changed

    for p in paths:
        if p.endswith(".py"):
            tree = ast.parse((ROOT / p).read_text())
            add_refs([tree], _late_aliases(tree, stems))
    keep = {p: set() for p in late}
    while True:
        changed = False
        for p, tree in trees.items():
            body = tree.body
            aliases = _late_aliases(tree, stems)
            want = set(roots[p])
            while True:
                kept = [i for i, n in enumerate(body) if _bound(n) & want or
                        (i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)) or
                        (isinstance(n, ast.ImportFrom) and n.module == "__future__") or
                        (want and isinstance(n, ast.Expr) and "sys.path" in ast.unparse(n))]
                refs = {n.id for i in kept for n in ast.walk(body[i]) if isinstance(n, ast.Name)}
                if refs <= want:
                    break
                want |= refs
            kept_set = set(kept) if roots[p] else set()
            if kept_set != keep[p]:
                keep[p] = kept_set
                changed = True
            if add_refs([body[i] for i in kept_set], aliases):
                changed = True
        if not changed:
            break
    out = {}
    for p, tree in trees.items():
        if not roots[p]:
            continue
        src = git("show", f"HEAD:{p}").split("\n")
        parts = []
        for i in sorted(keep[p]):
            n = tree.body[i]
            start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
            parts.append("\n".join(src[start - 1:n.end_lineno]))
        out[p] = "\n".join(parts) + "\n"
    return out


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
    rlist = out.parent / (out.name + ".replace")
    rlist.write_text(REPLACE_TEXT)
    subprocess.run([sys.executable, "-m", "git_filter_repo", "--paths-from-file", str(plist), "--force", "--quiet",
                    "--message-callback", MESSAGE_CALLBACK, "--replace-text", str(rlist)],
                   cwd=out, check=True)
    plist.unlink()
    rlist.unlink()

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
    shaken = shake_late(paths, late)
    for p, src in shaken.items():
        (out / p).parent.mkdir(parents=True, exist_ok=True)
        (out / p).write_text(neutral_module(src))
    late = list(shaken)
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
