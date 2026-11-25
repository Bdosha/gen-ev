import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
HISTORY_CSV = RESULTS / "history.csv"


def read_history():
    with HISTORY_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def series_by_generation(rows, config):
    # для каждого поколения: min, среднее и max лучшего f по запускам
    buckets = {}
    for row in rows:
        if row["config"] != config:
            continue
        gen = int(row["generation"])
        buckets.setdefault(gen, []).append(float(row["best_f"]))
    gens = sorted(buckets)
    mins, means, maxs = [], [], []
    for gen in gens:
        values = np.array(buckets[gen], dtype=float)
        mins.append(values.min())
        means.append(values.mean())
        maxs.append(values.max())
    return np.array(gens), np.array(mins), np.array(means), np.array(maxs)


def series_by_evals(rows, config):
    buckets = {}
    for row in rows:
        if row["config"] != config:
            continue
        n = int(row["n_evals"])
        buckets.setdefault(n, []).append(float(row["best_f"]))
    xs = sorted(buckets)
    means = [float(np.mean(buckets[x])) for x in xs]
    return np.array(xs), np.array(means)


def plot_generations(rows, config, title, out_path):
    gens, mins, means, maxs = series_by_generation(rows, config)
    plt.figure(figsize=(8, 4.5))
    plt.plot(gens, mins, label="минимум по запускам")
    plt.plot(gens, means, label="среднее по запускам")
    plt.plot(gens, maxs, label="максимум по запускам")
    plt.yscale("log")
    plt.xlabel("поколение")
    plt.ylabel("лучшее f")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("saved", out_path)


def plot_compare(rows, out_path):
    plt.figure(figsize=(8, 4.5))
    for config, label in (
        ("pop40", "ГА, популяция 40"),
        ("pop120", "ГА, популяция 120"),
        ("random", "случайные точки"),
    ):
        xs, means = series_by_evals(rows, config)
        plt.plot(xs, means, label=label)
    plt.yscale("log")
    plt.xlabel("число вызовов f")
    plt.ylabel("среднее лучшее f")
    plt.title("ГА и случайный поиск, 20 запусков")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("saved", out_path)


def main():
    if not HISTORY_CSV.exists():
        print("нет файла", HISTORY_CSV, file=sys.stderr)
        print("сначала: python lab01/run.py --all", file=sys.stderr)
        raise SystemExit(1)
    rows = read_history()
    plot_generations(
        rows,
        "pop40",
        "Популяция 40, лучшее f по поколениям",
        RESULTS / "ga_pop40.png",
    )
    plot_generations(
        rows,
        "pop120",
        "Популяция 120, лучшее f по поколениям",
        RESULTS / "ga_pop120.png",
    )
    plot_compare(rows, RESULTS / "ga_vs_random.png")


if __name__ == "__main__":
    main()
