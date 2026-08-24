namespace = "coding"
namespace_directive_rank = 1  # plan 28 Phase 3: lower = higher priority when the pump's budget cuts
namespace_directives = (
    "before writing new code, check the file and its module for something that already does it — "
    "reuse it rather than add a parallel version\n"
    "change only what is explicitly requested\n"
    "match the existing code style; introduce no new patterns\n"
    "do not over-engineer; add no speculative abstraction\n"
    "when a change alters how a value is produced, stored, or read, find every place that reads it "
    "and confirm each one still works with the new shape — a change is not finished until its "
    "consumers are\n"
    "derive what counts as a valid input from what the code that consumes it actually requires, not "
    "just from its surface format — a value can have the right shape and still be missing what a "
    "later step depends on\n"
    "when a change adds two or more pieces that call into each other, re-read them together once "
    "written — the boundary between them is where a change most often breaks quietly"
)
