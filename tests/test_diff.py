from agent.diff import DiffLine, build_diff, render_diff


# ---------------------------------------------------------------------------
# build_diff
# ---------------------------------------------------------------------------

class TestBuildDiff:
    def test_add_only(self):
        lines = build_diff("", "hello\nworld\n")
        kinds = [dl.kind for dl in lines]
        assert "add" in kinds
        assert "del" not in kinds

    def test_del_only(self):
        lines = build_diff("hello\nworld\n", "")
        kinds = [dl.kind for dl in lines]
        assert "del" in kinds
        assert "add" not in kinds

    def test_mixed(self):
        lines = build_diff("foo\nbar\n", "foo\nbaz\n")
        kinds = [dl.kind for dl in lines]
        assert "del" in kinds
        assert "add" in kinds

    def test_no_diff(self):
        lines = build_diff("same\n", "same\n")
        assert lines == []

    def test_hunk_header_present(self):
        lines = build_diff("a\n", "b\n")
        assert any(dl.kind == "header" for dl in lines)

    def test_file_header_lines_stripped(self):
        lines = build_diff("a\n", "b\n")
        assert not any(dl.text.startswith("---") or dl.text.startswith("+++") for dl in lines)

    def test_context_lines_present(self):
        old = "ctx1\nctx2\nctx3\nchange\nctx4\nctx5\nctx6\n"
        new = "ctx1\nctx2\nctx3\nchanged\nctx4\nctx5\nctx6\n"
        lines = build_diff(old, new)
        assert any(dl.kind == "context" for dl in lines)

    def test_cap_boundary(self):
        old = "\n".join(f"line{i}" for i in range(100)) + "\n"
        new = "\n".join(f"X{i}" for i in range(100)) + "\n"
        lines = build_diff(old, new, cap=10)
        changed = sum(1 for dl in lines if dl.kind in ("add", "del"))
        assert changed <= 10
        texts = [dl.text for dl in lines]
        assert any("more lines" in t for t in texts)

    def test_empty_to_new_file(self):
        lines = build_diff("", "new content\n")
        assert any(dl.kind == "add" for dl in lines)

    def test_text_prefix_correct(self):
        lines = build_diff("old\n", "new\n")
        add_lines = [dl for dl in lines if dl.kind == "add"]
        del_lines = [dl for dl in lines if dl.kind == "del"]
        assert all(dl.text.startswith("+") for dl in add_lines)
        assert all(dl.text.startswith("-") for dl in del_lines)


# ---------------------------------------------------------------------------
# render_diff
# ---------------------------------------------------------------------------

class TestRenderDiff:
    def test_returns_rich_text(self):
        from rich.text import Text
        lines = build_diff("old\n", "new\n")
        result = render_diff(lines)
        assert isinstance(result, Text)

    def test_empty_lines(self):
        from rich.text import Text
        result = render_diff([])
        assert isinstance(result, Text)

    def test_add_line_has_green_bg(self):
        dl = DiffLine(kind="add", text="+foo")
        result = render_diff([dl])
        # Rich stores spans; check the style string contains green bg color
        plain = result.plain
        assert "+foo" in plain

    def test_del_line_has_red_bg(self):
        dl = DiffLine(kind="del", text="-foo")
        result = render_diff([dl])
        plain = result.plain
        assert "-foo" in plain
