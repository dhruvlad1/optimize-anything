"""Fetch and verify the pinned upstream target; never edit the checkout."""
from __future__ import annotations

import inspect
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(__file__).resolve().parent
TARGET = PACKAGE / "target"
WORK = PACKAGE / "work"
PIN = "3ac3586"
URL = "https://github.com/grantjenks/python-sortedcontainers.git"
TEST_IGNORES = ["tests/benchmark.py", "tests/benchmark_plot.py", "tests/benchmark_sorteddict.py",
                "tests/benchmark_sortedlist.py", "tests/benchmark_sortedset.py",
                "tests/plot_lengths_histogram_add.py", "tests/plot_lengths_histogram_delitem.py"]


def _run(args, cwd=None, timeout=120):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True, timeout=timeout)


def _rmtree_onexc(function, path, exc):
    """Clear Windows read-only attributes and retry the failed removal."""
    if not isinstance(exc, PermissionError):
        raise exc
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IWRITE | stat.S_IREAD)
    function(path)


def prepare_work_copy():
    if not TARGET.is_dir():
        raise FileNotFoundError(f"target checkout missing: expected pinned checkout at {TARGET}; setup will not run Git")
    if WORK.exists():
        shutil.rmtree(WORK, onexc=_rmtree_onexc)
    ignored = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(TARGET, WORK, ignore=ignored)
    return WORK


def _read_head_without_git():
    """Read the local Git metadata directly; does not invoke Git or write target."""
    git_dir = TARGET / ".git"
    if git_dir.is_file():
        git_dir = TARGET / git_dir.read_text(encoding="utf-8").strip().split(":", 1)[1].strip()
    head_path = git_dir / "HEAD"
    head = head_path.read_text(encoding="ascii").strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(None, 1)[1]
    ref_path = git_dir / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="ascii").strip()
    packed = git_dir / "packed-refs"
    if packed.exists():
        for line in packed.read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[1] == ref:
                return fields[0]
    raise RuntimeError(f"cannot resolve target HEAD reference {ref} without Git")


def _coverage_statement_count(source):
    try:
        from coverage import Coverage
    except ImportError:
        return None
    try:
        cov = Coverage(data_file=None, source=[str(source)])
        return sum(len(cov._analyze(str(path)).statements) for path in source.rglob("*.py"))
    except Exception as exc:
        print(f"coverage statement count unavailable: {exc}", flush=True)
        return None


def main():
    if not TARGET.exists():
        raise FileNotFoundError(f"pinned target is not present at {TARGET}; this script does not clone or run Git")
    commit = _read_head_without_git()
    if not commit.startswith(PIN):
        raise RuntimeError(f"wrong target commit: expected {PIN}, got {commit[:12]}")
    src = TARGET / "src" / "sortedcontainers"
    lines = sum(sum(bool(line.strip()) for line in p.read_text(encoding="utf-8").splitlines())
                for p in src.rglob("*.py"))
    work = prepare_work_copy()
    work_src = work / "src"
    coverage_statements = _coverage_statement_count(work_src / "sortedcontainers")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(work_src) + os.pathsep + env.get("PYTHONPATH", "")
    collect_args = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=",
                    "--doctest-modules"]
    for ignored in TEST_IGNORES:
        collect_args.extend(["--ignore", ignored])
    collect_args.append("tests")
    collected = subprocess.run(collect_args, cwd=work, env=env, capture_output=True, text=True, timeout=180)
    if collected.returncode:
        raise RuntimeError(f"pytest collection failed in work copy ({collected.returncode}):\n{collected.stdout}\n{collected.stderr}")
    test_count = sum(1 for line in collected.stdout.splitlines() if "::" in line)
    print(f"rmtree signature: {inspect.signature(shutil.rmtree)}", flush=True)
    print(f"commit={commit[:7]}; nonblank_source_lines={lines}; coverage_statements={coverage_statements if coverage_statements is not None else 'not available'}; tests={test_count}", flush=True)
    if (work / ".git").exists():
        raise AssertionError("work copy unexpectedly contains .git")
    print(f"work_copy={work}; excluded .git and cache artifacts; setup complete", flush=True)


if __name__ == "__main__":
    main()
