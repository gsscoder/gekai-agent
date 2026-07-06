# Concepts

What makes Gekai different — conceptually, independent of current implementation status.

## The harness is the multiplier

The core bet: a well-built harness — tight scoping, precise file location, structured prompts,
checkpointed execution — lets a smaller, cheaper model produce results comparable to, or close
to, a stronger model used without one. Model strength is one input; how the model is orchestrated
is the other, and orchestration is what Gekai is actually optimizing. If the bet holds, the
ceiling on what's achievable stops being "which model can you afford" and becomes "how good is
the harness around it." Every other concept below is in service of this one.

## Surgical means precise, not small

The instinct is to call a tool "surgical" because it limits *how much* it touches. Gekai's
position: scope size is not the safety property that matters. A one-line edit in the wrong
place is worse than a mechanical rename across fifty files. What makes an intervention surgical
is whether it's *correct*, not whether it's *small*.

The precision mechanism is **verification of the change itself** — a check that runs on what was
actually created or modified, not on how many files were touched before anything happened. Plan 27
gives this a concrete home: a post-planning-only verify agent (a future `fact-checker`, or a
mechanical check until one exists) is attached to a plan step when its estimated complexity clears
a threshold — never assigned by prompt decomposition, only inserted afterward as a preventive
gate. On failure it triggers one repair-and-reverify pass; a second failure halts the plan in
place and reports which step failed — completed steps' work stays, nothing rolls back. The
complexity *metric* itself (what "clears the threshold" means, precisely) is still an open design
point.

## Checkpoint-oriented, not autonomous-run-oriented

Gekai is built around keeping a human in the loop, not long unsupervised execution. Every turn is
expected to produce something a human looks at before the next one starts. This shapes
everything downstream: subagents run cold with no carried context (fire-and-forget per turn,
by design), and diff/event output exists specifically to give the human something concrete to
check at each step.

The planner can decompose one request into a multi-step plan — several different specialists run
in order, each dispatched in turn by the fixed interpreter — but this is still transparent,
auto-run, single-turn machinery, not autonomous multi-turn operation: there is no interactive
plan-mode approval gate, no revert-on-failure, and no resume of a stopped plan. A failing step
stops the whole plan in place (completed steps' work stays, nothing rolls back) and reports which
step failed; the human still reviews everything that happened in one sitting before deciding what
to do next. The tool is not optimized for "give it a task and come back later" — it's optimized
for "watch it work in small verifiable increments," even when those increments are now chained.

## Routing is a guard, not a classifier

Two cheap guards, not one classifier trying to do everything: `Gate` decides chit-chat vs. act;
`Estimator` decides — for an act turn with no specific agent named — trivial vs. mutate. Neither
attempts to understand intent beyond its single fork, and neither ever names or reasons about a
specific specialist. That reasoning is reserved for the one place expensive enough to afford it:
the **planner**, which runs only on the mutation path (plan 27). Keeping the guards this narrow
keeps the common cases (chit-chat, a small read/edit) cheap, fast, and predictable — infrastructure
for dispatch, not a second opinion on what the user wants.

This supersedes plans 25 and 26. `delegate`-in-main — a tool the core model could elect to call
mid-turn — put the orchestration loop's control flow inside the model's own turn-by-turn
judgement, which is exactly the model-dependent variable the harness thesis exists to remove
(see [The harness is the multiplier](#the-harness-is-the-multiplier)): switching to a stronger
model would only mask the problem, not prove the harness. Plan 27's fix is to compile the
step-policy — execute, verify, repair-then-reverify, halt-on-second-failure — once, into a fixed,
engineered interpreter that knows no agent by name or role. Only the planner, which *does* know
the roster and their interactions, produces the data (a flat list of steps with optional
verify/repair agents) that the interpreter walks. Prose can no longer name a specialist either:
"use code-expert to do X" is read for its intent, not as a routing directive — a subagent is
summoned explicitly only via `/agent-x`, rejected at the command layer (never by the model
name-checking prose) if unknown.

## One persistent agent, many borrowed roles

There is exactly one long-lived agent identity (`Harness`) that the user always talks to.
Subagents are not separate tools or processes — they're scoped roles the same system plays for
a single turn, with their own restricted tool/permission view, then discarded. This avoids the
common multi-agent-framework trap of fragmenting identity and context across a fleet of
independent actors; Gekai stays one thing that can temporarily specialize, not a dispatcher
handing work to strangers.

## Each pipeline stage does exactly one job, and fails loud

Locate finds files. Rewrite attributes them into the request. Neither stage absorbs the
other's responsibility, and neither silently degrades on failure — a broken locate or a broken
rewrite blocks the turn rather than guessing. The cost of this is occasional hard stops; the
benefit is that when the agent acts, every upstream step that fed it context is known-good,
not best-effort.
