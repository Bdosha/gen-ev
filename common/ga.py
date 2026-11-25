import numpy as np


def clip_to_bounds(x, low, high):
    # координату за границей возвращаем на границу
    return np.clip(x, low, high)


def tournament(pop, fit, k, rng):
    # k случайных особей, берём ту, у которой f меньше
    idx = rng.integers(0, len(pop), size=k)
    winner = idx[np.argmin(fit[idx])]
    return pop[winner].copy()


def crossover(p1, p2, rng):
    # ребёнок = a * родитель1 + (1 - a) * родитель2
    a = float(rng.random())
    c1 = a * p1 + (1.0 - a) * p2
    c2 = (1.0 - a) * p1 + a * p2
    return c1, c2


def mutate(x, prob, sigma, low, high, rng):
    # с вероятностью prob к координате прибавляем нормальный шум
    x = np.array(x, dtype=float, copy=True)
    mask = rng.random(x.shape[0]) < prob
    noise = rng.normal(0.0, 1.0, size=x.shape[0]) * sigma
    x[mask] += noise[mask]
    return clip_to_bounds(x, low, high)


def run_ga(
    func,
    low,
    high,
    pop_size,
    eval_budget,
    tournament_size,
    crossover_prob,
    mutation_prob,
    mutation_coef,
    seed,
):
    rng = np.random.default_rng(seed)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    dim = int(low.shape[0])
    # один и тот же масштаб шума по каждой координате: coef * (верх - низ)
    sigma = mutation_coef * (high - low)

    # начальная популяция сразу внутри границ
    pop = rng.uniform(low, high, size=(pop_size, dim))
    fit = np.array([func(ind) for ind in pop], dtype=float)
    evals = pop_size

    best_i = int(np.argmin(fit))
    best_x = pop[best_i].copy()
    best_f = float(fit[best_i])
    # история: номер поколения, сколько раз вызвали f, лучшее f
    history = [(0, evals, best_f)]
    generation = 0

    while evals < eval_budget:
        generation += 1
        # в новом поколении одна особь — прошлая лучшая, её не пересчитываем
        slots = min(pop_size - 1, eval_budget - evals)
        children = []
        while len(children) < slots:
            p1 = tournament(pop, fit, tournament_size, rng)
            p2 = tournament(pop, fit, tournament_size, rng)
            if rng.random() < crossover_prob:
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = mutate(c1, mutation_prob, sigma, low, high, rng)
            c2 = mutate(c2, mutation_prob, sigma, low, high, rng)
            children.append(c1)
            if len(children) < slots:
                children.append(c2)

        children = np.asarray(children, dtype=float)
        child_fit = np.array([func(ind) for ind in children], dtype=float)
        evals += len(children)

        pop = np.vstack([best_x.reshape(1, -1), children])
        fit = np.concatenate([[best_f], child_fit])

        i = int(np.argmin(fit))
        if fit[i] < best_f:
            best_f = float(fit[i])
            best_x = pop[i].copy()
        history.append((generation, evals, best_f))

    return {
        "best_x": best_x,
        "best_f": best_f,
        "n_evals": evals,
        "history": history,
    }
