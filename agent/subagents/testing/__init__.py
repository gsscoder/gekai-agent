namespace = "testing"
namespace_directive_rank = 2  # plan 28 Phase 3: lower = higher priority when the pump's budget cuts
namespace_directives = (
    "match the project's existing test conventions — framework, assertion style, layout, "
    "fixtures, runner; if no existing tests or conventions exist, adopt the ecosystem default "
    "(pytest, conventional layout) and proceed\n"
    "every test traces to evidence — spec, signature, comment, or observable behavior; "
    "never invent a requirement\n"
    "assertions are specific and falsifiable — assert exact values, not mere truthiness\n"
    "never silently assume — document assumptions and flag gaps for the human\n"
    "when the repository already has a test runner set up, a logic change ships with a test that "
    "would fail without the fix; never introduce a new test framework or runner where none exists — "
    "report the gap instead"
)
