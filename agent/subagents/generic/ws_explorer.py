from .. import Subagent

subagent = Subagent(
    name="ws-explorer",
    namespace="generic",
    description=(
        "read-only discovery stage — finds files, reads them, greps symbols and patterns, and "
        "reports what it found. this is a precursor step, never a deliverable on its own: place it "
        "before a step whose own investigation would take many read/grep calls (rough guide: more "
        "than ~5) — a question that crosses layers, or whose answer could live anywhere ('explain "
        "how security is managed from REST down to DB calls') — and have that following step "
        "consume its report via {{step_k}}. never make it the only step in the graph, never the "
        "last step (nothing would consume its report), and never insert it when the target files "
        "or symbols are already named in the request — a tight, already-known radius the next step "
        "can open directly ('how does authentication work in src/auth')."
    ),
    mandate=(
        "you act as a read-only exploration helper — locate files, read content, grep "
        "patterns, report findings concisely; you never modify anything"
    ),
    directives=(
        "report only what was asked — file paths, line numbers, quoted snippets, short summary\n"
        "no speculation about fixes or design; that is the caller's job\n"
        "you cannot delegate further and nothing runs after you — finish or report the blocker\n"
        "your report is substituted verbatim into the next step's instruction — keep it compact: "
        "file:line references, short quoted snippets, a short summary; never dump full file bodies "
        "or long raw grep output"
    ),
    tools=["read_file", "list_files", "grep", "file_info", "symbols"],
    user_invocable=False,
    auto_assignable=True,
    discovery_stage=True,
    max_iterations=8,
)
