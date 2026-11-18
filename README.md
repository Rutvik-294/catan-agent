# Catan Agent — PIMC for the catanatron engine

A **Parallel Implicit Monte-Carlo (PIMC)** agent built on the
[catanatron](https://github.com/bcollazo/catanatron) engine for a Catan
hackathon.

## How it decides

For each decision the agent samples **K complete worlds** (`obs.sample_world`)
— different hypotheses about opponents' hidden cards and the dev deck — and
scores every playable action by executing it in each world and reading the
catanatron value function. It then picks the action with the best **mean
score** across worlds.

Unlike the stock bot (which reasons over a single world and is blind to hidden
information), the multi-world average *sees* the opponent's possible hidden
victory points distributed across worlds, and a terminal-reachability
correction prices in "the opponent is already at 10".

## The clock is the real constraint

About 1 s of CPU per game, ~12.5 ms per decision — so the search stays 1-ply
and K is chosen from the clock:

- expensive prompts with many equivalent actions are pruned deterministically
  (robber spots keep the distinct victim options, maritime trades keep only
  trades that complete a build) so more worlds fit in the budget;
- the per-decision budget is a slice of what `obs.cpu_left` reports, sized to
  stay inside the 12.5 ms increment once past the opening;
- opening placement decisions get a wider budget (300 ms base).

## Files

- `agent.py` — the agent (samples worlds, scores actions, manages the time budget)
- `test_agent.py` — quick sanity tests

## Usage

Drop the agent into the hackathon kit's player interface:

```python
class Agent:
    def decide(self, obs, playable_actions) -> int:
        ...
```
