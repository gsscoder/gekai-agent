from __future__ import annotations

from agent.pipeline.blast_radius import count_blast_areas, evaluate_blast_radius_gate


# ---------------------------------------------------------------------------
# count_blast_areas
# ---------------------------------------------------------------------------

def test_count_single_dir() -> None:
    paths = ["agent/tools/shell.py", "agent/tools/list.py"]
    assert count_blast_areas(paths) == 1


def test_count_nested_collapses_to_parent() -> None:
    # agent/tools is ancestor of agent/tools/helpers — collapses to 1
    paths = ["agent/tools/shell.py", "agent/tools/helpers/fs.py"]
    assert count_blast_areas(paths) == 1


def test_count_sibling_dirs() -> None:
    paths = ["agent/tools/shell.py", "agent/helpers/sanitizer.py"]
    assert count_blast_areas(paths) == 2


def test_count_four_files_two_dirs() -> None:
    paths = [
        "agent/tools/read.py",
        "agent/tools/list.py",
        "agent/helpers/sanitizer.py",
        "agent/helpers/compress.py",
    ]
    assert count_blast_areas(paths) == 2


def test_count_root_file_pulls_in_subdirs() -> None:
    # agent is ancestor of agent/tools — collapses to 1
    paths = ["agent/foo.py", "agent/tools/bar.py"]
    assert count_blast_areas(paths) == 1


def test_count_empty() -> None:
    assert count_blast_areas([]) == 0


def test_count_single_file() -> None:
    assert count_blast_areas(["agent/tools/shell.py"]) == 1


# ---------------------------------------------------------------------------
# evaluate_blast_radius_gate
# ---------------------------------------------------------------------------

def _entries(paths: list[str]) -> list[tuple[str, list[str]]]:
    return [(p, []) for p in paths]


def test_gate_passes_at_limit() -> None:
    paths = [f"pkg{i}/file.py" for i in range(5)]
    rejected, reason = evaluate_blast_radius_gate(_entries(paths), limit=5)
    assert not rejected
    assert reason is None


def test_gate_rejects_over_limit() -> None:
    paths = [f"pkg{i}/file.py" for i in range(6)]
    rejected, reason = evaluate_blast_radius_gate(_entries(paths), limit=5)
    assert rejected
    assert reason is not None
    assert "6" in reason
    assert "5" in reason


def test_gate_passes_empty_entries() -> None:
    rejected, reason = evaluate_blast_radius_gate([], limit=5)
    assert not rejected


# ---------------------------------------------------------------------------
# _CODE_EXTENSIONS filtering
# ---------------------------------------------------------------------------

def test_count_ignores_config_files() -> None:
    # pyproject.toml and package.json excluded; only .py counts
    paths = ["agent/tools/shell.py", "pyproject.toml", "package.json"]
    assert count_blast_areas(paths) == 1


def test_count_config_only_is_zero() -> None:
    paths = ["pyproject.toml", "setup.cfg", "package.json", ".gitignore"]
    assert count_blast_areas(paths) == 0


def test_count_jsx_counts() -> None:
    paths = ["src/components/Button.jsx", "src/components/Modal.jsx"]
    assert count_blast_areas(paths) == 1


def test_count_jsx_and_ts_two_dirs() -> None:
    paths = ["src/components/Button.jsx", "src/hooks/useAuth.ts"]
    assert count_blast_areas(paths) == 2


def test_count_md_and_yaml_excluded() -> None:
    paths = ["docs/guide.md", ".github/workflows/ci.yml", "src/main.py"]
    assert count_blast_areas(paths) == 1


def test_count_vue_and_svelte_count() -> None:
    paths = ["src/views/Home.vue", "src/components/Nav.svelte"]
    assert count_blast_areas(paths) == 2


def test_gate_passes_when_only_configs_touched() -> None:
    # config-only change: 0 code areas → always passes regardless of limit
    entries = [("pyproject.toml", []), ("setup.cfg", []), ("README.md", [])]
    rejected, reason = evaluate_blast_radius_gate(entries, limit=1)
    assert not rejected


def test_gate_mixed_code_and_config_counts_only_code() -> None:
    # 3 config files + 2 code files in same dir → 1 area → passes limit=2
    entries = [
        ("pyproject.toml", []),
        ("package.json", []),
        ("tsconfig.json", []),
        ("src/main.ts", []),
        ("src/utils.ts", []),
    ]
    rejected, _ = evaluate_blast_radius_gate(entries, limit=2)
    assert not rejected


def test_gate_reason_lists_areas() -> None:
    paths = ["agent/tools/a.py", "agent/helpers/b.py", "frontend/c.tsx"]
    rejected, reason = evaluate_blast_radius_gate(_entries(paths), limit=2)
    assert rejected
    assert reason is not None
    # all three survivor dirs named in reason
    assert "agent/tools" in reason or "agent\\tools" in reason
    assert "agent/helpers" in reason or "agent\\helpers" in reason
    assert "frontend" in reason
