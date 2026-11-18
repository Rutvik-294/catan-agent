"""Smoke test: PimcAgent vs baseline catanatron bots. Prints win rates + decision timing."""
import collections
import sys
import time

sys.path.insert(0, "/home/rutvik/catan_agent")

import agent as A  # noqa: E402
from catanatron.game import Game  # noqa: E402
from catanatron.models.player import Color  # noqa: E402
from catanatron.players.search import VictoryPointPlayer  # noqa: E402
from catanatron.players.weighted_random import WeightedRandomPlayer  # noqa: E402

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def run(opp_cls, label):
    wins = collections.Counter()
    stats = collections.Counter()
    t0 = time.time()
    for i in range(N_GAMES):
        me_seat = i % 2
        colors = [Color.RED, Color.BLUE]

        class P(A.PimcAgent):
            def decide(self, game, playable_actions=None):
                obs = A._view_from(game, self.color)
                t1 = time.time()
                a = A.decide(obs)
                stats["tot"] += time.time() - t1
                stats["n"] += 1
                stats["mx"] = max(stats["mx"], time.time() - t1)
                return a

        players = sorted([P(colors[me_seat]), opp_cls(colors[1 - me_seat])],
                         key=lambda p: p.color != Color.RED)
        game = Game(players=players)
        w = game.play()
        wins[str(w) if w is not None else "nobody"] += 1
    print(f"{label}: {dict(wins)}  decisions={stats['n']} "
          f"avg={stats['tot'] / max(1, stats['n']):.2f}s max={stats['mx']:.2f}s "
          f"total={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    run(WeightedRandomPlayer, "vs WeightedRandomPlayer")
    run(VictoryPointPlayer, "vs VictoryPointPlayer")
