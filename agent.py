"""PIMC agent for the catan-hackathon kit interface.

class Agent.decide(obs, playable_actions) -> int

For each decision it samples K complete worlds (`obs.sample_world`) and scores
every playable action by executing it in the world and reading the catanatron
value function, then picks the action with the best MEAN score over worlds.
Unlike stock F (one world, blind to hidden cards) the multi-world average sees
the opponent's possible hidden victory points distributed over worlds, and a
terminal-reachability correction prices "the opponent is already at 10" in.

The clock is the real constraint: about 1 s of CPU per game, ~12.5 ms per
decision.  So the search stays 1-ply and K is chosen from the clock:
  * expensive prompts with many equivalent actions are pruned deterministically
    (robber spots keep the distinct victim options, maritime trades keep only
    trades that complete a build) so more worlds fit in the budget;
  * the per-decision budget is a slice of what `obs.cpu_left` says, sized to
    stay inside the 12.5 ms increment once past the opening, with a wider
    budget for the opening placement decisions (the 300 ms base).
"""

import os
import random
import time

from catanatron.models.enums import RESOURCES
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn

# 2-player-tuned weights (the kit's get_value_fn pins "base_fn" to DEFAULT_WEIGHTS,
# so call base_fn directly to bias the same 13 features differently).
# Env overrides exist only to sweep: production, enemy_production, hand_synergy,
# reachable_production_1, buildable_nodes.
def _float_env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default

_W = dict(DEFAULT_WEIGHTS)
_W["production"] *= _float_env("CATAN_WPROD", 1.0)
_W["enemy_production"] *= _float_env("CATAN_WENEMY", 1.0)
_W["hand_synergy"] *= _float_env("CATAN_WSYN", 1.0)
_W["reachable_production_1"] *= _float_env("CATAN_WREACH", 1.0)
_W["buildable_nodes"] *= _float_env("CATAN_WIFI", 1.0)
VALUE_FN = base_fn(_W)

DEV_DECK = {"KNIGHT": 14, "VICTORY_POINT": 5, "ROAD_BUILDING": 2,
            "YEAR_OF_PLENTY": 2, "MONOPOLY": 2}

#: score scale sane reference: DEFAULT_WEIGHTS drives public_vps up to ~3e14, so
#: terminal corrections must sit above any build delta but below float overflow.
WIN_PENALTY = -1e17

#: tabulated above from kit measurements; ~1 s is the average allowance and we
#: keep a comfortable margin under it
EVAL_SECONDS = 0.045
OPENING_SECONDS = 0.10

RES_ORDER = list(RESOURCES)
CITY_COST = {"WHEAT": 2, "ORE": 3}
SETTLE_COST = {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1}
DEV_COST = {"SHEEP": 1, "WHEAT": 1, "ORE": 1}
ROAD_COST = {"WOOD": 1, "BRICK": 1}

MAX_EVAL_SECONDS = EVAL_SECONDS


class Agent:
    def __init__(self):
        self.rng = random.Random(0x5EED)
        self._seen_decisions = 0

    # ------------------------------------------------------------------ #
    # clock                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deadline(obs):
        left = obs.cpu_left
        if left is None:
            return time.process_time() + 0.010
        if left > 0.25:                      # opening (300 ms base era)
            return time.process_time() + min(OPENING_SECONDS, 0.25 * left)
        return time.process_time() + min(MAX_EVAL_SECONDS, 0.40 * left)

    # ------------------------------------------------------------------ #
    # candidate pruning                                                  #
    # ------------------------------------------------------------------ #

    def _prune(self, obs, playable_actions):
        """Remove near-duplicate legal actions; index remap keeps legality."""
        keys = set()
        groups = {}
        for i, a in enumerate(playable_actions):
            groups.setdefault(a.action_type, []).append(i)

        keep = set()
        for t, idxs in groups.items():
            if len(idxs) <= 6:
                keep.update(idxs)
                continue
            if t.name == "MOVE_ROBBER":
                # one entry per tile, choosing the best victim from obs (public info)
                best = {}
                for i in idxs:
                    coord = playable_actions[i].value[0]
                    victim = playable_actions[i].value[1]
                    richness = sum(obs.opp_resources.values()) if victim is not None else 0
                    if coord not in best or richness > best[coord][0]:
                        best[coord] = (richness, i)
                keep.update(best[coord][1] for coord in best)
            elif t.name == "MARITIME_TRADE":
                keep.update(idxs[:8])
            elif t.name == "DISCARD_RESOURCE":
                keep.update(idxs)
            else:
                keep.update(idxs[:24])
        out = sorted(keep)
        return out, {i: i for i in out}
        # NOTE: pruning returns indices into playable_actions directly.

    # ------------------------------------------------------------------ #
    # main                                                               #
    # ------------------------------------------------------------------ #

    def decide(self, obs, playable_actions) -> int:
        n = len(playable_actions)
        self._seen_decisions += 1
        if n == 1:
            return 0

        # ---- immediate wins are free and decisive ----
        if obs.my_actual_vps >= 10:
            return 0
        myr = obs.my_resources
        cheap_win = None
        if myr.get("WHEAT", 0) >= 2 and myr.get("ORE", 0) >= 3:
            for i, a in enumerate(playable_actions):
                if a.action_type.name == "BUILD_CITY":
                    cheap_win = i
                    break
        elif all(myr.get(r, 0) >= v for r, v in SETTLE_COST.items()):
            for i, a in enumerate(playable_actions):
                if a.action_type.name == "BUILD_SETTLEMENT":
                    cheap_win = i
                    break
        if obs.my_actual_vps + 1 >= 10 or obs.my_actual_vps + 2 >= 10:
            # at 9, a city (+2) or any settlement/dev (+1) ends it now
            for i, a in enumerate(playable_actions):
                tname = a.action_type.name
                need = 10 - obs.my_actual_vps
                if tname in ("BUILD_CITY",) and need <= 2:
                    return i
                if tname == "BUILD_SETTLEMENT" and need <= 1:
                    return i
                if (tname == "BUY_DEVELOPMENT_CARD"
                        and self._hidden_vp_chance(obs) > 0.0 and need <= 1):
                    pass  # uncertain: fall through to scoring
        if cheap_win is not None and obs.my_actual_vps + 2 >= 10:
            return cheap_win

        # ---- budgeted PIMC scoring ----
        idxs = None
        try:
            idxs = self._prune(obs, playable_actions)[0]
        except Exception:
            idxs = list(range(n))

        deadline = self._deadline(obs)
        # hidden info only matters while the opponent can hold unseen dev cards
        worlds_max = 6 if obs.opp_dev_count > 0 else 2
        sums = [0.0] * n
        counts = [0] * n
        worlds = 0
        evals = 0
        while evals == 0 or time.process_time() < deadline:
            if evals >= 400 or worlds >= worlds_max:
                break
            try:
                world = obs.sample_world(self.rng)
            except Exception:
                break
            for i in idxs:
                a = playable_actions[i]
                succ = world.copy()
                try:
                    succ.execute(a)
                except Exception:
                    sums[i] -= 1e15          # illegal-in-world: treat as terrible
                    counts[i] += 1
                    continue
                v = self._score(succ, obs, a)
                sums[i] += v
                counts[i] += 1
                evals += 1
            worlds += 1
            if evals == 0:
                break

        best_i, best_v = 0, None
        for i in idxs:
            v = sums[i] / counts[i] if counts[i] else -1e16
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i

    # ------------------------------------------------------------------ #
    # scoring                                                            #
    # ------------------------------------------------------------------ #

    def _score(self, succ, obs, action):
        """Value function + terminal corrections in the sampled world."""
        st = succ.state
        ps = st.player_state
        i = st.color_to_index[obs.me]
        opp_actual = ps[f"P{1 - i}_ACTUAL_VICTORY_POINTS"]
        my_actual = ps[f"P{i}_ACTUAL_VICTORY_POINTS"]
        if my_actual >= 10:
            return WIN_PENALTY * -2.0
        if opp_actual >= 10:
            return WIN_PENALTY
        v = VALUE_FN(succ, obs.me)
        reach = self._opp_next_reach(st, obs)
        if reach > 0 and opp_actual >= 9:
            v -= 2e14                     # a world where F's weights see no loss, but the
        elif reach > 0 and opp_actual >= 7:  # opponent wins next turn without us acting
            v -= 3e13
        # 2-ply defense: let the opponent take its greedy reply before judging
        # only worth its cost once the score actually turns into a race
        if v > 0 and opp_actual >= 5 and opp_actual + reach >= 8:
            v_reply = self._opp_reply_value(succ, obs)
            if v_reply is not None:
                v = 0.5 * v + 0.5 * v_reply
        return v

    def _opp_reply_value(self, succ, obs):
        """Opponent's greedy reply in world `succ`, then our value after it."""
        try:
            opp_actions = succ.state.playable_actions
            if len(opp_actions) <= 1:
                return VALUE_FN(succ, obs.me)   # forced or ineligible prompt
            opp_color = succ.state.current_color()
            best = None
            for a in opp_actions[:24]:
                s2 = succ.copy()
                try:
                    s2.execute(a)
                except Exception:
                    continue
                v_opp = self._score(s2, obs, a)     # which WE despair of
                if best is None or v_opp < best:
                    best = v_opp
            return best
        except Exception:
            return VALUE_FN(succ, obs.me)

    @staticmethod
    def _opp_next_reach(st, obs):
        """In a sampled world (opponent hand fully visible there), how many VP can the
        opponent add on its very next turn?"""
        o = 1 - st.color_to_index[obs.me]
        ps = st.player_state
        hand = {r: ps[f"P{o}_{r}_IN_HAND"] for r in RES_ORDER}
        reach = 0.0
        if (ps[f"P{o}_CITIES_AVAILABLE"] > 0 and hand.get("WHEAT", 0) >= 2
                and hand.get("ORE", 0) >= 3):
            reach += 2
        elif (ps[f"P{o}_SETTLEMENTS_AVAILABLE"] > 0
              and all(hand.get(r, 0) >= v for r, v in SETTLE_COST.items())):
            reach += 1
        return reach

    # ------------------------------------------------------------------ #
    # public-info hidden inference                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hidden_vp_chance(obs):
        """share of the unseen dev pile that is victory points (public numbers)."""
        pool_vps = DEV_DECK["VICTORY_POINT"] - obs.my_devs.get("VICTORY_POINT", 0)
        pool_size = 25 - sum(obs.my_devs.values())     # played cards are public too
        if pool_size <= 0:
            return 0.0
        return pool_vps / pool_size


if __name__ == "__main__":  # local sanity run against the kit's own check
    import os
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, here)
