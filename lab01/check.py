import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ga import run_ga
from discus import discus


def main():
    # в нуле функция равна нулю
    got = discus(np.zeros(5))
    if got != 0.0:
        raise SystemExit(f"discus(0) = {got}, ожидал 0")

    low = [-5.0] * 5
    high = [5.0] * 5
    kwargs = dict(
        func=discus,
        low=low,
        high=high,
        pop_size=30,
        eval_budget=200,
        tournament_size=3,
        crossover_prob=0.9,
        mutation_prob=0.2,
        mutation_coef=0.1,
    )
    a = run_ga(seed=1, **kwargs)
    b = run_ga(seed=1, **kwargs)
    if a["n_evals"] != 200 or b["n_evals"] != 200:
        raise SystemExit(f"число вызовов {a['n_evals']}, ожидал 200")
    if a["best_f"] != b["best_f"] or not np.allclose(a["best_x"], b["best_x"]):
        raise SystemExit("два запуска с одним seed дали разный ответ")
    if np.any(a["best_x"] < -5) or np.any(a["best_x"] > 5):
        raise SystemExit("лучшая точка вышла за границы")
    print("check ok", a["best_f"])


if __name__ == "__main__":
    main()
