import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.ga import run_ga
from discus import discus

RESULTS = Path(__file__).resolve().parent / "results"
RUNS_CSV = RESULTS / "runs.csv"
HISTORY_CSV = RESULTS / "history.csv"

RUN_FIELDS = ["config", "seed", "best_f", "best_x", "n_evals", "seconds"]
HIST_FIELDS = ["config", "seed", "generation", "n_evals", "best_f"]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    dim = int(cfg["dimension"])
    cfg["low_vec"] = [float(cfg["low"])] * dim
    cfg["high_vec"] = [float(cfg["high"])] * dim
    cfg["seeds"] = [int(s) for s in cfg["seeds"]]
    return cfg


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replace_config_rows(path, fieldnames, config_name, new_rows):
    old = [row for row in read_csv(path) if row["config"] != config_name]
    write_csv(path, fieldnames, old + new_rows)


def format_x(x):
    return " ".join(f"{v:.8g}" for v in x)


def run_one_ga(cfg, seed):
    t0 = time.perf_counter()
    result = run_ga(
        func=discus,
        low=cfg["low_vec"],
        high=cfg["high_vec"],
        pop_size=int(cfg["pop_size"]),
        eval_budget=int(cfg["eval_budget"]),
        tournament_size=int(cfg["tournament_size"]),
        crossover_prob=float(cfg["crossover_prob"]),
        mutation_prob=float(cfg["mutation_prob"]),
        mutation_coef=float(cfg["mutation_coef"]),
        seed=seed,
    )
    seconds = time.perf_counter() - t0
    run_row = {
        "config": cfg["name"],
        "seed": seed,
        "best_f": f"{result['best_f']:.10e}",
        "best_x": format_x(result["best_x"]),
        "n_evals": result["n_evals"],
        "seconds": f"{seconds:.4f}",
    }
    hist_rows = []
    for generation, n_evals, best_f in result["history"]:
        hist_rows.append(
            {
                "config": cfg["name"],
                "seed": seed,
                "generation": generation,
                "n_evals": n_evals,
                "best_f": f"{best_f:.10e}",
            }
        )
    return run_row, hist_rows


def random_search(cfg):
    # равномерные точки в тех же границах и с тем же числом вызовов f
    low = np.asarray(cfg["low_vec"], dtype=float)
    high = np.asarray(cfg["high_vec"], dtype=float)
    budget = int(cfg["eval_budget"])
    dim = low.shape[0]
    run_rows = []
    hist_rows = []
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        best_f = float("inf")
        best_x = None
        for n in range(1, budget + 1):
            x = rng.uniform(low, high, size=dim)
            value = discus(x)
            if value < best_f:
                best_f = value
                best_x = x.copy()
            # в файл пишем не каждую точку, а каждую 20-ю и последнюю
            if n % 20 == 0 or n == budget:
                hist_rows.append(
                    {
                        "config": "random",
                        "seed": seed,
                        "generation": "",
                        "n_evals": n,
                        "best_f": f"{best_f:.10e}",
                    }
                )
        seconds = time.perf_counter() - t0
        run_rows.append(
            {
                "config": "random",
                "seed": seed,
                "best_f": f"{best_f:.10e}",
                "best_x": format_x(best_x),
                "n_evals": budget,
                "seconds": f"{seconds:.4f}",
            }
        )
        print(f"random seed={seed} best_f={best_f:.6e} time={seconds:.3f}s")
    return run_rows, hist_rows


def run_config(cfg):
    run_rows = []
    hist_rows = []
    for seed in cfg["seeds"]:
        run_row, hist = run_one_ga(cfg, seed)
        run_rows.append(run_row)
        hist_rows.extend(hist)
        print(
            f"{cfg['name']} seed={seed} best_f={run_row['best_f']} "
            f"evals={run_row['n_evals']} time={run_row['seconds']}s"
        )
    return run_rows, hist_rows


def save_pair(config_name, run_rows, hist_rows):
    replace_config_rows(RUNS_CSV, RUN_FIELDS, config_name, run_rows)
    replace_config_rows(HISTORY_CSV, HIST_FIELDS, config_name, hist_rows)


def main():
    parser = argparse.ArgumentParser(description="Серия запусков для лабораторной 1")
    parser.add_argument("--config", help="JSON с параметрами одного запуска ГА")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Оба размера популяции и случайный поиск",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Только случайный поиск. Бюджет и seed берутся из --config или из config_base.json",
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent

    if args.all:
        for name in ("config_base.json", "config_pop120.json"):
            cfg = load_config(here / name)
            runs, hist = run_config(cfg)
            save_pair(cfg["name"], runs, hist)
        base = load_config(here / "config_base.json")
        runs, hist = random_search(base)
        save_pair("random", runs, hist)
        return

    if args.random:
        path = Path(args.config) if args.config else here / "config_base.json"
        cfg = load_config(path)
        runs, hist = random_search(cfg)
        save_pair("random", runs, hist)
        return

    if not args.config:
        parser.error("нужен --config или --all")
    cfg = load_config(args.config)
    runs, hist = run_config(cfg)
    save_pair(cfg["name"], runs, hist)


if __name__ == "__main__":
    main()
