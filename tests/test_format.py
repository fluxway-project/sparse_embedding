import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["--check", "--fix"])
@pytest.mark.parametrize("from_bazel", [False, True])
def test_format_only_selects_git_worktree_sources(tmp_path, mode, from_bazel):
    repository = tmp_path / "repository with spaces"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    scripts = repository / "scripts"
    scripts.mkdir()
    shutil.copyfile(PROJECT / "scripts/format.sh", scripts / "format.sh")
    (repository / ".gitignore").write_text("tmp/\n.venv/\nbuild/\n")
    files = ["source with spaces.cpp", "BUILD.bazel", "source.py"]
    for name in files:
        (repository / name).write_text("source marker\n")
    subprocess.run(
        ["git", "-C", str(repository), "add", "source with spaces.cpp"], check=True
    )
    for directory in ("tmp", ".venv", "build"):
        path = repository / directory
        path.mkdir()
        for name in files:
            (path / name).write_text("ignored marker\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "calls.jsonl"
    for name in ("clang-format-16", "buildifier", "ruff"):
        executable = tools / name
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['FORMAT_TEST_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv) + '\\n')\n"
        )
        executable.chmod(0o755)
    before = {
        path.relative_to(repository): path.read_bytes()
        for path in repository.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    subprocess.run(
        ["bash", str(scripts / "format.sh"), mode],
        cwd=tmp_path if from_bazel else repository,
        env={
            **os.environ,
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
            "FORMAT_TEST_LOG": str(log),
            "GIT_CEILING_DIRECTORIES": str(tmp_path.parent),
            **({"BUILD_WORKSPACE_DIRECTORY": str(repository)} if from_bazel else {}),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 4
    arguments = [argument for call in calls for argument in call[1:]]
    assert all(name in arguments for name in files)
    assert not any(
        argument.startswith(("tmp/", ".venv/", "build/")) for argument in arguments
    )
    if mode == "--check":
        assert "--dry-run" in calls[0]
        assert "-mode=check" in calls[1]
        assert "--check" in arguments
        assert "--fix" not in arguments
    assert all(
        (repository / path).read_bytes() == content for path, content in before.items()
    )
