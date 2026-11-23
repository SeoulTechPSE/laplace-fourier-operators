"""Anonymous code supplement for the TMLR submission of the (C) paper.

    python experiments/C/paper/tmlr/make_supplement.py RELEASE_DIR OUT.zip

RELEASE_DIR is a release build (`release/make_release.py`), so the follow-up study is already out. The zip holds
the shared package, its tests and the (C) experiments with every experiment module they import (transitively);
not the A / D experiments or any paper draft. Author, affiliation and repository are removed: LICENSE and
pyproject name "Anonymous authors", CITATION.cff and the release README are left out, and a short README is
written. The script fails if an identifying string remains.
"""
from __future__ import annotations

import ast
import re
import sys
import zipfile
from pathlib import Path

KEEP = ("src/lfo/", "tests/", "experiments/C/e1_audit/", "experiments/C/e5_probe/")
DROP = re.compile(r"(__pycache__|\.pyc$|/figures/.*\.pdf$)")
BANNED = re.compile(r"Kee-Youn|\bYoo\b|SeoulTech|kyyoo|seoultech|github\.com/SeoulTechPSE", re.I)
README = """# Code supplement (anonymous)

Code for "Do Laplace Neural Operators Learn Physical Poles? An Audit on Linear Systems and a Trust Rule for
Nonlinear Ones".

- `src/lfo/`: shared package (models, data generators, metrics); `pip install -e .`, tests: `pytest`.
- `experiments/C/e1_audit/`: linear audit (Sections 3-4, appendix). README: runs, results and every number.
- `experiments/C/e5_probe/`: nonlinear trust rule (Sections 5-8). README: the passes, pre-registrations
  (`results/*prereg*.md`, each committed before its test) and results.
- Other `experiments/` modules are imported by these scripts.

Python >= 3.10, PyTorch; baselines from `neuraloperator`.
"""


def imported(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def files(root: Path) -> list[Path]:
    tracked = [p.relative_to(root) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    keep = {p for p in tracked if str(p).startswith(KEEP) and not DROP.search(str(p))}
    modules = {}
    for p in tracked:
        if p.suffix == ".py" and str(p).startswith("experiments/"):
            modules.setdefault(p.stem, []).append(p)
    todo = [p for p in keep if p.suffix == ".py"]
    while todo:
        for name in imported(root / todo.pop()):
            for q in modules.get(name, []):
                if q not in keep:
                    keep.add(q)
                    todo.append(q)
    return sorted(keep)


def main():
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    keep = files(root)
    lic = re.sub(r"Copyright \(c\) (\d+) .*", r"Copyright (c) \1 Anonymous authors", (root / "LICENSE").read_text())
    pyp = re.sub(r"authors = \[.*\]", 'authors = [{ name = "Anonymous authors" }]', (root / "pyproject.toml").read_text())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", README)
        z.writestr("LICENSE", lic)
        z.writestr("pyproject.toml", pyp)
        for p in keep:
            data = (root / p).read_bytes()
            try:
                if BANNED.search(data.decode()):
                    sys.exit(f"identifying string in {p}")
            except UnicodeDecodeError:
                pass
            z.writestr(str(p), data)
    with zipfile.ZipFile(out) as z:               # final check of everything written, generated files included
        for name in z.namelist():
            try:
                if BANNED.search(z.read(name).decode()):
                    sys.exit(f"identifying string in {name}")
            except UnicodeDecodeError:
                pass
    print(f"{out}: {len(keep) + 3} files, {out.stat().st_size / 2**20:.1f} MB; outside experiments/C:",
          sorted({str(p) for p in keep if p.parts[0] == 'experiments' and p.parts[1] != 'C'}))


if __name__ == "__main__":
    main()
