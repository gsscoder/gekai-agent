# Concepts

What makes Gekai different — conceptually, independent of current implementation status.

## The harness is the multiplier

The core bet: a well-built harness — tight scoping, precise file location, structured prompts,
checkpointed execution — lets a smaller, cheaper model produce results comparable to, or close
to, a stronger model used without one. Model strength is one input; how the model is orchestrated
is the other, and orchestration is what Gekai is actually optimizing. If the bet holds, the
ceiling on what's achievable stops being "which model can you afford" and becomes "how good is
the harness around it." Every other concept below is in service of this one.

## The harness is a boundary, not an identity

"Harness" names *all* engineered, non-LLM, per-turn machinery — gate, estimator, sequencer,
interpreter, permission gate, event bridge, persistence, the turn loop itself — not any single
class or call. Main is not the harness; main is one configuration of the LLM loop the harness
deploys, the same way a subagent is another. The harness's public surface is a **touchpoint
registry**: every place it invokes a model is an enumerated entry (`gate`, `estimator`,
`sequencer`, `main-dispatch`, `subagent-dispatch`, `micro`), each with its own job and its own
tier. Adding a model interaction means adding a registry entry, not scattering a new client call
into whichever module happens to need one — the registry *is* the operational definition, not a
comment describing one.

This is also the answer to "when do we spawn a specialist?" A bare `hi` never reaches a specialist
— it's pure harness plus the model, nothing staged. Complexity earns more harness, not automatically
more agents: main can be pumped with domain expertise (see below) before the harness ever reaches
for a cold spawn. The harness scales with the turn; specialists are one of several tools it reaches
for, not the default shape of "more work."

## Surgical means precise, not small

The instinct is to call a tool "surgical" because it limits *how much* it touches. Gekai's
position: scope size is not the safety property that matters. A one-line edit in the wrong
place is worse than a mechanical rename across fifty files. What makes an intervention surgical
is whether it's *correct*, not whether it's *small*.

The precision mechanism is **verification of the change itself** — a check that runs on what was
actually created or modified, not on how many files were touched before anything happened. Plan 27
gives this a concrete home: a post-planning-only verify agent (a future `fact-checker`, or a
mechanical check until one exists) is attached to a task graph step when its estimated complexity
clears a threshold — never assigned by prompt decomposition, only inserted afterward as a
preventive gate. On failure it triggers one repair-and-reverify pass; a second failure halts the
task graph in place and reports which step failed — completed steps' work stays, nothing rolls
back. The complexity *metric* itself (what "clears the threshold" means, precisely) is still an
open design point.

## Checkpoint-oriented, not autonomous-run-oriented

Gekai is built around keeping a human in the loop, not long unsupervised execution. Every turn is
expected to produce something a human looks at before the next one starts. This shapes
everything downstream: subagents run cold with no carried context (fire-and-forget per turn,
by design), and diff/event output exists specifically to give the human something concrete to
check at each step. Each step's instruction does carry a mechanical `<request_summary>` block —
the task graph's overall gist plus that step's boundary against its siblings — but this is
request-level framing derived from the current task graph, not accumulated cross-turn state, so
the "no carried context" principle still holds turn to turn.

The sequencer can decompose one request into a multi-step task graph — several different
specialists run in order, each dispatched in turn by the fixed interpreter — but this is still
transparent, auto-run, single-turn machinery, not autonomous multi-turn operation: there is no
interactive approval gate, no revert-on-failure, and no resume of a halted graph. A failing step
stops the whole graph in place (completed steps' work stays, nothing rolls back) and reports which
step failed; the human still reviews everything that happened in one sitting before deciding what
to do next. The tool is not optimized for "give it a task and come back later" — it's optimized
for "watch it work in small verifiable increments," even when those increments are now chained.
`TaskGraph` is deliberately named ahead of its v1 shape: v1 is linear (edges implicit-sequential,
identical in behavior to a flat step list); conditional branching is a named, deliberately deferred
capability, not built — building it would put control flow back into generated data, reversing a
decision made for exactly the reasons in this section.

## Routing is a guard, not a classifier

Two cheap guards, not one classifier trying to do everything: `Gate` decides chit-chat vs. act;
`Estimator` decides — for an act turn with no specific agent named — trivial vs. mutate. Neither
attempts to understand intent beyond its single fork, and neither ever names or reasons about a
specific specialist. That reasoning is reserved for the one place expensive enough to afford it:
the **sequencer**, which runs only on the mutation path (plan 27). Keeping the guards this narrow
keeps the common cases (chit-chat, a small read/edit) cheap, fast, and predictable — infrastructure
for dispatch, not a second opinion on what the user wants.

This supersedes plans 25 and 26. `delegate`-in-main — a tool the core model could elect to call
mid-turn — put the orchestration loop's control flow inside the model's own turn-by-turn
judgement, which is exactly the model-dependent variable the harness thesis exists to remove
(see [The harness is the multiplier](#the-harness-is-the-multiplier)): switching to a stronger
model would only mask the problem, not prove the harness. Plan 27's fix is to compile the
step-policy — execute, verify, repair-then-reverify, halt-on-second-failure — once, into a fixed,
engineered interpreter that knows no agent by name or role. Only the sequencer (the `sequencer`
touchpoint), which *does* know the roster and their interactions, produces the data — a task
graph, v1 a flat list of steps with optional verify/repair agents — that the interpreter walks.
Prose can no longer name a specialist either:
"use code-expert to do X" is read for its intent, not as a routing directive — a subagent is
summoned explicitly only via `/agent-x`, rejected at the command layer (never by the model
name-checking prose) if unknown.

## One persistent agent, many borrowed roles

There is exactly one long-lived agent identity — **main** — that the user always talks to. The
harness is not that identity; it is the engineered machinery that deploys main and, when a turn
warrants it, spawns a specialist on main's behalf. Subagents are not separate tools or processes —
they're scoped roles the same system plays for a single turn, with their own restricted
tool/permission view, then discarded. This avoids the common multi-agent-framework trap of
fragmenting identity and context across a fleet of independent actors; Gekai stays one thing that
can temporarily specialize, not a dispatcher handing work to strangers.

## Model tiers are capability contracts, not models

A tier (`FAST`, `SUPP`, `CORE`) is a capability contract, not a model — the same model can occupy
every tier at a different operating point, or different tiers can hold entirely different models;
a component neither knows nor cares which. Global config binds each tier to a concrete
`(model, creds, url)`; per-model **suitability** metadata (`ok` / `warning` / `deprecated` per
tier) is advisory only, surfaced as a config-time warning and never a block — misconfiguring a
tier is the user's prerogative, not an error the harness polices.

Every component — a touchpoint, main, a subagent — declares a **configured default** operating
point plus a **limits space**: the tiers it may run at, each with an effort range, thinking legal
only inside `CORE`. The space is the entire mobility model; a component with one tier at one
effort (a future file-explorer's `tiers=[fast(low)]`) is simply non-scalable — no separate flag
needed. Configured efforts are fixed; only the harness moves a component, and only within its
declared space, decided **at assignment time**: is the default adequate for this work? If not,
move up or down, across tiers or within a tier's range, never by the model's own self-assessment
(that would rebuild the very router this architecture removed). A reasoning need force-promotes a
component to `core(low)` at minimum — but only if the component's space includes `core` at all;
otherwise it simply runs without thinking. The harness is the mover, never the moved: it carries no
scalability flag of its own, only a default binding for any touchpoint that doesn't declare one.

## Dynamic directives: main borrows expertise without borrowing a role

Domain directives (coding, testing, …) are ranked by how deeply they presuppose a role's
*mission* versus stating domain *craft*. Craft — "prefer explicit named parameters," "never use
mutable defaults" — is mission-free and safe to stack into main directly; mission — "only realign
existing tests, never add new cases" — is coherent only inside its specialist's remit and never
escapes. When the harness deploys main for a turn, it detects the turn's domains mechanically
(file extensions, manifests, keywords — no model call deciding the mix) and pumps main's prompt
with the escaping directives of those domains, budgeted so a multi-domain turn doesn't bloat the
prompt past what a smaller model can still follow.

This is the missing middle rung between "main alone" and "spawn a specialist": main borrows the
*expertise* a domain demands without borrowing the *role* a cold spawn would carry. Specialists
don't disappear — they keep what a directive can't confer: permission/tool scoping and
context isolation from the conversation. Dynamic directives replaces dispatch-for-expertise, not
dispatch-for-isolation.

## A linter for instruction files, not a guardrail

A project can hand Gekai standing instructions two ways — its own `GEKAI.md`, ingested verbatim
into every turn, or a foreign file like `AGENTS.md` that the user asks Gekai to read once, which
lands in message history like any other file read and is compaction fodder like any other file
read. Framings that would gate or filter either file were each considered and dropped: a modal
prompt (read-but-ignore / rewrite / cancel) fires on the very file the user wrote *for* Gekai and
teaches click-through; silently stripping directives before injection makes Gekai the only agent
that eats part of a file every other tool honors, with the false-positive cost landing on the user
as a missing instruction they cannot see; a mechanical regex prefilter was tried and shipped once,
then deleted — tested against a real sibling repo's `AGENTS.md`, it returned a false negative on a
genuine instruction file while matching this project's own `AGENTS.md` on subject matter alone
("AI coding agent" appearing in prose it was never meant to gate on). Every mechanical shortcut
this project has tried in place of a model turn has failed the same way: it reads *what a file is
about*, not *who a file addresses*. What ships instead is a single cheap, non-thinking LLM call
that asks the file one question — does it contain rules meant to influence an AI agent's behavior —
and tells the human, once, if the answer is yes. Never the model, and never a block.

This is proportionate specifically because the surface a stronger control would protect was never
prose to begin with: permissions are code (`agent/permissions.py`, `agent/settings.py`), and a
markdown line asking for `exec` cannot grant it. The honest scope of the feature is telling the
human that a file addresses an AI agent's behavior, and stopping there — no corpus, no comparison
against Gekai's own directives, no classification of *how* it disagrees. If the user ignores the
notice, whatever the file asks for runs for the whole session, every turn — the notice's
persistence (it is not a timed toast; it stays until the next verdict replaces it) is the only
mitigation, deliberately, because a stronger mechanism is the thing this design keeps refusing to
build. The same honesty shapes the fallback direction on a bad or failed answer: a false `NO`
(a missed warning) costs nothing but a skipped notice, while a false `YES` teaches the user to
ignore every later warning — so every unparseable output and every call failure degrades to the
silent, safe `NO`, never the reverse.

## Each pipeline stage does exactly one job, and fails loud

Locate finds files. Rewrite attributes them into the request. Neither stage absorbs the
other's responsibility, and neither silently degrades on failure — a broken locate or a broken
rewrite blocks the turn rather than guessing. The cost of this is occasional hard stops; the
benefit is that when the agent acts, every upstream step that fed it context is known-good,
not best-effort.
