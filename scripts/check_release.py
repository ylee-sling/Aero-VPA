# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 SLING AI Inc.
"""Release hygiene checks for the Aero-VPA repository.

Run from anywhere::

    python scripts/check_release.py              # static checks only (no third-party imports)
    python scripts/check_release.py --run-tests  # + ruff and the self-tests, in dependency order

Checks
------
1. Every ``*.py`` file that would be published starts with exactly the two lines of ``SPDX_HEADER``.
2. ``LICENSE`` is the unmodified GNU GPL v3 text (674 lines, canonical SHA-256 of the FSF file).
3. ``docs/README.md`` states that the preprint PDF is CC BY 4.0, (C) 2026 Yeonseok Lee, and not under the GPL;
   the PDF itself is present and is a PDF.
4. ``README.md`` carries the title, the DOI, the exact commercial-licensing sentence, the Mermaid diagram,
   the BibTeX entry and the licence split.
5. ``CITATION.cff`` carries the DOI, ``GPL-3.0-only``, the repository URL and a semantic version.
6. The publishable file set (what ``git add -A`` would commit; a filesystem walk with the ``.gitignore``
   patterns when there is no git repository) contains every expected file and no build or model artefacts,
   oversized files or absolute paths of a build machine.

Exit status is 0 if every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TITLE = (
    "Aero-VPA: Few-Step Flow-Matching Control over Non-Reconstructive JEPA Latents "
    "for Agile Flight on Edge Hardware"
)
DOI = "10.5281/zenodo.23633375"
REPO_URL = "https://github.com/ylee-sling/Aero-VPA"
SPDX_HEADER = ("# SPDX-License-Identifier: GPL-3.0-only", "# Copyright (C) 2026 SLING AI Inc.")
COMMERCIAL_NOTICE = (
    "For commercial integration, proprietary licensing, or defense adaptation, contact ylee@sling.ai.kr."
)
# SHA-256 of https://www.gnu.org/licenses/gpl-3.0.txt (GPL v3, 29 June 2007), 674 lines, LF line endings.
GPL3_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
GPL3_LINES = 674

# Self-test commands in dependency order (a script with options runs as "python <file> <options>").
SELF_TESTS = ("config.py", "perception.py", "predictor.py", "selector.py", "solver.py", "pipeline.py",
              "checkpoint.py", "sim/expert.py", "dataset.py", "train_stage1.py --self-test",
              "cache_latents.py --self-test", "train_stage2.py --self-test", "calibrate_sigma.py --self-test",
              "diagnose_stage1.py --self-test", "probe_depth.py --self-test", "clearance.py",
              "sim/fly_policy.py --self-test", "sim/summarize_flights.py --self-test", "envelope_check.py --self-test",
              "sim/show_trace.py --self-test", "input_sensitivity.py --self-test", "sim/dagger.py")
LINT_COMMAND = ("ruff", "check", "--select", "F,E9,B,PLE,PLW", ".")
EXPECTED_FILES = (
    *(command.split()[0] for command in SELF_TESTS),
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    ".gitattributes",
    "docs/README.md",
    "docs/preprint_261026.pdf",
    "scripts/check_release.py",
    "scripts/release.sh",
    ".github/workflows/selftest.yml",
    "inspect_hdf5.py",
    "sim/collect_expert_dataset.py",
    "sim/collect_worlds.sh",
    "sim/evaluate_worlds.sh",
    "sim/dagger_round.sh",
    "sim/make_world.py",
    "sim/run_headless_sitl.sh",
    "sim/models/aero_vpa_rgbd/model.config",
    "sim/models/aero_vpa_rgbd/model.sdf",
    "sim/models/x500_vpa/model.config",
    "sim/models/x500_vpa/model.sdf",
)
PREPRINT_PDF = "docs/preprint_261026.pdf"

# Never published, whatever .gitignore says (a force-added file is still caught).
FORBIDDEN_PATTERNS = (
    "*.pyc", "*.pyo", "__pycache__/*", "*/__pycache__/*", ".ruff_cache/*", ".pytest_cache/*",
    "*.egg-info/*", "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.onnx", "*.engine", "*.plan",
    "*.zip", "*.tar.gz", ".DS_Store", "*/.DS_Store", "Thumbs.db", ".env", "*.pem", "*.key",
)
MAX_FILE_BYTES = 5 * 1024 * 1024                      # the preprint PDF is ~0.4 MB
BUILD_MACHINE_PATHS = re.compile(rb"/home/claude|/mnt/user-data|/tmp/claude-|/root/\.")
SECRET_PATTERNS = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".cff", ".yml", ".yaml", ".sh", ".toml", ".cfg", ".sdf", ".config", ""}


class Report:
    """Collects check results and prints them as ``PASS``/``FAIL`` lines."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        if ok:
            self.passes += 1
            print(f"  PASS  {label}")
        else:
            self.failures.append(label)
            print(f"  FAIL  {label}" + (f": {detail}" if detail else ""))
        return ok


def _read_text(rel: str) -> str | None:
    path = ROOT / rel
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _gitignore_patterns() -> list[str]:
    text = _read_text(".gitignore") or ""
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def _ignored(rel: str, patterns: list[str]) -> bool:
    """Small subset of gitignore semantics: ``name``, ``*.ext`` and ``dir/`` patterns (enough for this repo)."""
    parts = rel.split("/")
    for pat in patterns:
        if pat.endswith("/"):
            if pat.rstrip("/") in parts[:-1]:
                return True
        elif "/" not in pat.strip("/"):
            if any(fnmatch.fnmatch(part, pat) for part in parts):
                return True
        elif fnmatch.fnmatch(rel, pat.lstrip("/")):
            return True
    return False


def publishable_files() -> tuple[list[str], str]:
    """Files that ``git add -A`` would commit, or a .gitignore-filtered walk without git."""
    if (ROOT / ".git").exists() and shutil.which("git"):
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            check=True, capture_output=True,
        ).stdout.decode("utf-8")
        files = sorted({f for f in out.split("\0") if f and (ROOT / f).exists()})
        return files, "git index + untracked, honouring .gitignore"
    patterns = _gitignore_patterns()
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            rel = Path(dirpath, name).relative_to(ROOT).as_posix()
            if not _ignored(rel, patterns):
                files.append(rel)
    return sorted(files), "filesystem walk with .gitignore patterns (no git repository)"


def check_headers(report: Report, files: list[str]) -> None:
    print("[1] SPDX headers on Python sources")
    py_files = [f for f in files if f.endswith(".py")]
    report.check(len(py_files) >= len(SELF_TESTS) + 1, f"{len(py_files)} Python files found")
    for rel in py_files:
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()[:2]
        report.check(tuple(lines) == SPDX_HEADER, f"header of {rel}", f"first lines are {lines!r}")


def check_license(report: Report) -> None:
    print("[2] LICENSE is the unmodified GPL v3")
    path = ROOT / "LICENSE"
    if not report.check(path.is_file(), "LICENSE present"):
        return
    raw = path.read_bytes()
    report.check(b"\r" not in raw, "LICENSE has LF line endings")
    n_lines = raw.count(b"\n")
    report.check(n_lines == GPL3_LINES, f"LICENSE has {GPL3_LINES} lines", f"{n_lines} lines")
    digest = hashlib.sha256(raw).hexdigest()
    report.check(digest == GPL3_SHA256, "LICENSE SHA-256 matches the FSF text", digest)


def check_docs(report: Report) -> None:
    print("[3] docs/: preprint and its CC BY 4.0 notice")
    text = _read_text("docs/README.md")
    if not report.check(text is not None, "docs/README.md present"):
        return
    flat = " ".join(text.split())
    for needle in (
        "Copyright (C) 2026 Yeonseok Lee",
        "Creative Commons Attribution 4.0 International",
        "CC BY 4.0",
        "https://creativecommons.org/licenses/by/4.0/legalcode",
        "**not** covered by the GNU General Public License",
        DOI,
        "preprint_261026.pdf",
    ):
        report.check(needle in flat, f"docs/README.md mentions {needle!r}")
    pdf = ROOT / PREPRINT_PDF
    if report.check(pdf.is_file(), f"{PREPRINT_PDF} present"):
        report.check(pdf.read_bytes()[:5] == b"%PDF-", f"{PREPRINT_PDF} is a PDF")


def check_readme(report: Report) -> None:
    print("[4] README.md")
    text = _read_text("README.md")
    if not report.check(text is not None, "README.md present"):
        return
    report.check(text.splitlines()[0] == f"# {TITLE}", "README title")
    report.check(COMMERCIAL_NOTICE in text, "exact commercial-licensing sentence")
    report.check(f"https://doi.org/{DOI}" in text, "DOI link")
    report.check(f"doi       = {{{DOI}}}" in text and "@misc{lee2026aerovpa," in text, "BibTeX entry with DOI")
    report.check(f"url       = {{{REPO_URL}}}" in text, "BibTeX url is the plain repository URL")
    report.check(text.count("```") % 2 == 0, "code fences are balanced")
    report.check("```mermaid" in text, "Mermaid architecture diagram")
    report.check("CC BY 4.0" in text and "GPL-3.0-only" in text, "licence split stated")
    for badge in ("GPL--3.0--only", "zenodo." + DOI.split("zenodo.")[1], "Python-3.10", "PyTorch-2.14", "ROS%202",
                  "Jetson%20Orin%20Nano"):
        report.check(badge in text, f"badge {badge!r}")
    report.check(all(f"python {name}" in text for name in SELF_TESTS), f"quick start lists the {len(SELF_TESTS)} self-tests")


def check_citation(report: Report) -> None:
    print("[5] CITATION.cff")
    text = _read_text("CITATION.cff")
    if not report.check(text is not None, "CITATION.cff present"):
        return
    for needle in (
        "cff-version: 1.2.0",
        f'doi: "{DOI}"',
        "license: GPL-3.0-only",
        f'repository-code: "{REPO_URL}"',
        f'title: "{TITLE}"',
    ):
        report.check(needle in text, f"CITATION.cff has {needle!r}")
    version = citation_version(text)
    report.check(version is not None, "CITATION.cff has a semantic version", "missing 'version: \"X.Y.Z\"'")


def citation_version(text: str | None = None) -> str | None:
    text = _read_text("CITATION.cff") if text is None else text
    match = re.search(r'^version:\s*"?(\d+\.\d+\.\d+)"?\s*$', text or "", re.MULTILINE)
    return match.group(1) if match else None


def check_tree(report: Report, files: list[str], source: str) -> None:
    print(f"[6] publishable file set ({len(files)} files; {source})")
    present = set(files)
    for rel in EXPECTED_FILES:
        report.check(rel in present, f"{rel} included")
    bad = [f for f in files if any(fnmatch.fnmatch(f, p) for p in FORBIDDEN_PATTERNS)]
    report.check(not bad, "no build, model or archive artefacts", ", ".join(bad))
    big = [f for f in files if (ROOT / f).stat().st_size > MAX_FILE_BYTES]
    report.check(not big, f"no file larger than {MAX_FILE_BYTES // 2**20} MiB", ", ".join(big))
    leaks, secrets, crlf = [], [], []
    for rel in files:
        if Path(rel).suffix not in TEXT_SUFFIXES or rel == "scripts/check_release.py":
            continue
        data = (ROOT / rel).read_bytes()
        if BUILD_MACHINE_PATHS.search(data):
            leaks.append(rel)
        if SECRET_PATTERNS.search(data):
            secrets.append(rel)
        if b"\r\n" in data:
            crlf.append(rel)
    report.check(not leaks, "no absolute paths of a build machine", ", ".join(leaks))
    report.check(not secrets, "no private keys or access tokens", ", ".join(secrets))
    report.check(not crlf, "text files use LF line endings", ", ".join(crlf))


def run_tests(report: Report) -> None:
    print("[7] lint and self-tests")
    report.check(sys.version_info >= (3, 10), "Python >= 3.10", sys.version.split()[0])
    ruff = shutil.which("ruff")
    if report.check(ruff is not None, "ruff available", "pip install -r requirements-dev.txt"):
        lint = subprocess.run([ruff, *LINT_COMMAND[1:]], cwd=ROOT, capture_output=True, text=True, check=False)
        report.check(lint.returncode == 0, " ".join(LINT_COMMAND), (lint.stdout + lint.stderr).strip()[-2000:])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    for name in SELF_TESTS:
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, *name.split()], cwd=ROOT, capture_output=True, text=True, env=env, check=False
        )
        seconds = time.perf_counter() - start
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-25:]
        ok = report.check(proc.returncode == 0, f"python {name} ({seconds:.1f} s)", "\n" + "\n".join(tail))
        summary = proc.stdout.strip().splitlines()[-1:]                     # the test's own last line
        if ok and summary:
            print(f"        {summary[0]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-tests", action="store_true", help="also run ruff and the self-tests")
    args = parser.parse_args(argv)

    print(f"Aero-VPA release check: {ROOT}")
    report = Report()
    files, source = publishable_files()
    check_headers(report, files)
    check_license(report)
    check_docs(report)
    check_readme(report)
    check_citation(report)
    check_tree(report, files, source)
    if args.run_tests:
        run_tests(report)

    print()
    if report.failures:
        print(f"FAILED: {len(report.failures)} of {report.passes + len(report.failures)} checks")
        for label in report.failures:
            print(f"  - {label}")
        return 1
    print(f"OK: all {report.passes} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
