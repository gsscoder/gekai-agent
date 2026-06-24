# Concepts

What makes Gekai different — conceptually, independent of current implementation status.

## Surgical means precise, not small

The instinct is to call a tool "surgical" because it limits *how much* it touches. Gekai's
position: scope size is not the safety property that matters. A one-line edit in the wrong
place is worse than a mechanical rename across fifty files. What makes an intervention surgical
is whether it's *correct*, not whether it's *small*.

The blast-radius gate (area-count limit before a subagent runs) exists, but it is a coarse,
cheap pre-filter — not the feature that defines the tool. It catches runaway mis-scoping early;
it says nothing about whether the change itself was right. Treating it as the core safety
mechanism overstates what it does and risks blocking legitimate wide-but-correct changes.

The real precision mechanism is meant to be **verification of the change itself** — a check
that runs on what was actually created or modified, not on how many files were touched before
anything happened. This is the direction, not yet a settled design: a non-invocable subagent
that reviews a create/update once it's non-trivial, as a second pass distinct from the gate.
Undecided still: whether it blocks/reverts on failure or only reports into the human checkpoint.

## Checkpoint-oriented, not autonomous-run-oriented

Gekai is built around keeping a human in the loop, not long unsupervised execution. Every turn is
expected to produce something a human looks at before the next one starts. This shapes
everything downstream: subagents run cold with no carried context (fire-and-forget per turn,
by design), and diff/event output exists specifically to give the human something concrete to
check at each step.

The router can decompose one request into a multi-step plan — several different specialists run
in order, each on its own locate/gate/rewrite/dispatch pass — but this is still transparent,
auto-run, single-turn machinery, not autonomous multi-turn operation: there is no interactive
plan-mode approval gate, no revert-on-failure, and no resume of a stopped plan. A failing step
stops the whole plan in place (completed steps' work stays, nothing rolls back) and reports which
step failed; the human still reviews everything that happened in one sitting before deciding what
to do next. The tool is not optimized for "give it a task and come back later" — it's optimized
for "watch it work in small verifiable increments," even when those increments are now chained.

## Routing is a guard, not a classifier

`Router` makes exactly one decision per turn: does this stay with the main agent, get rejected,
get answered without touching the codebase, or get handed to a specialist. It does not attempt
to understand intent beyond that single fork. Keeping the router's job this narrow keeps it
cheap, fast, and predictable to reason about — it is infrastructure for dispatch, not a second
opinion on what the user wants.

## One persistent agent, many borrowed roles

There is exactly one long-lived agent identity (`Harness`) that the user always talks to.
Subagents are not separate tools or processes — they're scoped roles the same system plays for
a single turn, with their own restricted tool/permission view, then discarded. This avoids the
common multi-agent-framework trap of fragmenting identity and context across a fleet of
independent actors; Gekai stays one thing that can temporarily specialize, not a dispatcher
handing work to strangers.

## Each pipeline stage does exactly one job, and fails loud

Locate finds files. Rewrite attributes them into the request. The gate bounds area. None of
these stages absorb another's responsibility, and none of them silently degrade on failure —
a broken locate or a broken rewrite blocks the turn rather than guessing. The cost of this is
occasional hard stops; the benefit is that when the agent acts, every upstream step that fed it
context is known-good, not best-effort.
