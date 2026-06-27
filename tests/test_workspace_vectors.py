from __future__ import annotations

import sqlite_vec

import agent.workspace.embed as embed_module
from agent.workspace.db import (
    ensure,
    find_candidates,
    find_hybrid,
    find_semantic,
    save_chunk_vectors,
    save_findings,
)


def _insert_file_with_vec(conn, tmp_path, rel_path: str, keywords: list[str], vec: list[float]) -> None:
    (tmp_path / rel_path).write_text("content")
    save_findings(conn, tmp_path, [(rel_path, keywords)])
    fid = conn.execute("SELECT id FROM files WHERE path = ?", (rel_path,)).fetchone()[0]
    conn.execute(
        "INSERT INTO file_chunks(file_id, chunk_idx, vector) VALUES (?, ?, ?)",
        (fid, 0, sqlite_vec.serialize_float32(vec)),
    )
    conn.commit()


def test_knn_roundtrip(tmp_path, monkeypatch) -> None:
    conn = ensure(tmp_path)
    dim = 384
    vec_a = [1.0] + [0.0] * (dim - 1)
    vec_b = [0.0] + [1.0] + [0.0] * (dim - 2)
    _insert_file_with_vec(conn, tmp_path, "a.py", ["alpha"], vec_a)
    _insert_file_with_vec(conn, tmp_path, "b.py", ["beta"], vec_b)

    # Query closest to a.py
    monkeypatch.setattr(embed_module, "embed_texts", lambda texts: [vec_a])
    results = find_semantic(conn, tmp_path, "query", k=5)
    conn.close()

    paths = [p for p, _ in results]
    assert "a.py" in paths
    assert paths.index("a.py") < paths.index("b.py")


def test_find_hybrid_combines_both(tmp_path, monkeypatch) -> None:
    conn = ensure(tmp_path)
    dim = 384
    # a.py: wins lexical (exact keyword match), mediocre vector
    # b.py: wins semantic (close vector), no keyword match
    vec_a = [0.5] + [0.0] * (dim - 1)
    vec_b = [1.0] + [0.0] * (dim - 1)
    _insert_file_with_vec(conn, tmp_path, "a.py", ["the_keyword"], vec_a)
    _insert_file_with_vec(conn, tmp_path, "b.py", ["other"], vec_b)

    # Query vector closest to b.py
    query_vec = [1.0] + [0.0] * (dim - 1)
    monkeypatch.setattr(embed_module, "embed_texts", lambda texts: [query_vec])

    results = find_hybrid(conn, tmp_path, ["the_keyword"], "query", k=5)
    conn.close()

    # Both files should appear (each wins one retriever)
    assert "a.py" in results
    assert "b.py" in results


def test_find_hybrid_graceful_when_no_chunks(tmp_path, monkeypatch) -> None:
    conn = ensure(tmp_path)
    (tmp_path / "x.py").write_text("def x(): pass")
    save_findings(conn, tmp_path, [("x.py", ["x", "py"])])

    # No vectors inserted — find_semantic returns empty, lexical still works
    monkeypatch.setattr(embed_module, "embed_texts", lambda texts: [[0.0] * 384])
    results = find_hybrid(conn, tmp_path, ["x"], "find x", k=5)
    conn.close()

    assert "x.py" in results


def test_save_chunk_vectors_replaces_on_reindex(tmp_path) -> None:
    conn = ensure(tmp_path)
    (tmp_path / "f.py").write_text("old")
    save_findings(conn, tmp_path, [("f.py", ["f"])])

    dim = 384
    vec1 = [1.0] + [0.0] * (dim - 1)
    vec2 = [0.5] + [0.0] * (dim - 1)

    save_chunk_vectors(conn, "f.py", ["chunk0"], [vec1])
    count_before = conn.execute("SELECT COUNT(*) FROM file_chunks").fetchone()[0]

    save_chunk_vectors(conn, "f.py", ["new_chunk0"], [vec2])
    count_after = conn.execute("SELECT COUNT(*) FROM file_chunks").fetchone()[0]

    conn.close()
    assert count_before == 1
    assert count_after == 1  # replaced, not accumulated
