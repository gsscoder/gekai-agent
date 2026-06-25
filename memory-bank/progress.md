# Project Progress

## Status
Incomplete Alpha version

## Current Focus
Implementing missing harness and improving existing one

## Recent Changes
- The pipeline matured into its current shape: per-subagent tool allowlists, split subagent identities
- The router gained multi-step plan execution that sequences a turn across multiple specialists with budget-exhaustion salvage
- Blast-radius gate was removed since the planner itself scopes subagent turns directly
- Removed the router's non-English rejection policy, leaving language handling entirely to the model