from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from deap import algorithms, tools

from ga_optimizer.config import DEFAULT_GA_CONFIG
from ga_optimizer.ga.data import decode_chromosome, load_ga_input
from ga_optimizer.ga.evaluator import build_xgbrf_evaluator
from ga_optimizer.ga.toolbox import build_toolbox


def _resolve_population_size(requested_population_size: int, n_features: int) -> int:
    if requested_population_size > 0:
        return int(requested_population_size)
    cfg = DEFAULT_GA_CONFIG.toolbox
    proposed = int(np.ceil(n_features * float(cfg.dynamic_population_ratio)))
    return max(int(cfg.dynamic_population_min), min(int(cfg.dynamic_population_max), proposed))


def _evaluate_invalid_individuals(
    toolbox: Any,
    population: list[Any],
    fitness_cache: dict[tuple[int, ...], float],
    *,
    verbose: bool = False,
    phase_label: str = "evaluation",
) -> int:
    invalid = [individual for individual in population if not individual.fitness.valid]
    total = len(invalid)
    if total == 0:
        if verbose:
            print(f"[GA] {phase_label}: no individuals to evaluate.", flush=True)
        return 0

    cache_hits = 0
    duplicate_hits = 0
    to_evaluate: list[tuple[Any, tuple[int, ...]]] = []
    duplicate_waiting: list[tuple[Any, tuple[int, ...]]] = []
    seen_uncached: set[tuple[int, ...]] = set()
    for individual in invalid:
        chromosome_key = tuple(int(gene) for gene in individual)
        cached = fitness_cache.get(chromosome_key)
        if cached is not None:
            individual.fitness.values = (float(cached),)
            cache_hits += 1
            continue
        if chromosome_key in seen_uncached:
            duplicate_hits += 1
            duplicate_waiting.append((individual, chromosome_key))
            continue
        seen_uncached.add(chromosome_key)
        to_evaluate.append((individual, chromosome_key))

    uncached_total = len(to_evaluate)
    if verbose:
        print(
            f"[GA] {phase_label}: total_invalid={total}, "
            f"cache_hits={cache_hits}, duplicate_hits={duplicate_hits}, "
            f"to_evaluate={uncached_total}",
            flush=True,
        )

    if uncached_total == 0:
        return total

    started_at = time.perf_counter()
    progress_stride = max(1, uncached_total // 10)
    map_fn = getattr(toolbox, "map", map)
    uncached_individuals = [individual for individual, _key in to_evaluate]
    heartbeat_stop: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    if verbose:
        heartbeat_interval_seconds = 15.0

        def _heartbeat() -> None:
            while heartbeat_stop is not None and not heartbeat_stop.wait(
                heartbeat_interval_seconds
            ):
                elapsed = time.perf_counter() - started_at
                print(
                    f"[GA] {phase_label}: still evaluating "
                    f"{uncached_total} uncached individuals "
                    f"(elapsed {elapsed:.1f}s)",
                    flush=True,
                )

        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

    try:
        fitness_iter = map_fn(toolbox.evaluate, uncached_individuals)
        for idx, ((individual, chromosome_key), fitness) in enumerate(
            zip(to_evaluate, fitness_iter), start=1
        ):
            individual.fitness.values = fitness
            fitness_cache[chromosome_key] = float(fitness[0])
            if verbose and (idx == uncached_total or idx % progress_stride == 0):
                elapsed = time.perf_counter() - started_at
                print(
                    f"[GA] {phase_label}: {idx}/{uncached_total} uncached complete "
                    f"(elapsed {elapsed:.1f}s)",
                    flush=True,
                )
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)

    for individual, chromosome_key in duplicate_waiting:
        individual.fitness.values = (float(fitness_cache[chromosome_key]),)

    return total


def _population_stats(stats: tools.Statistics, population: list[Any]) -> dict[str, float]:
    compiled = stats.compile(population)
    return {
        "avg": float(compiled["avg"]),
        "std": float(compiled["std"]),
        "min": float(compiled["min"]),
        "max": float(compiled["max"]),
    }


def _population_best_score(population: list[Any]) -> float:
    return float(max(individual.fitness.values[0] for individual in population))


def run_ga(
    csv_path: str,
    label_column: str,
    task: Literal["classification", "regression"],
    scoring: str,
    output_dir: str,
    is_time_series: bool = False,
    time_series_train_window: Optional[int] = DEFAULT_GA_CONFIG.evaluator.time_series_train_window,
    time_series_test_window: Optional[int] = DEFAULT_GA_CONFIG.evaluator.time_series_test_window,
    time_series_step_window: Optional[int] = DEFAULT_GA_CONFIG.evaluator.time_series_step_window,
    time_series_gap: int = DEFAULT_GA_CONFIG.evaluator.time_series_gap,
    max_time_series_splits: Optional[int] = DEFAULT_GA_CONFIG.evaluator.max_time_series_splits,
    n_generations: int = DEFAULT_GA_CONFIG.runner.n_generations,
    population_size: int = DEFAULT_GA_CONFIG.toolbox.population_size,
    cx_prob: float = DEFAULT_GA_CONFIG.toolbox.cx_prob,
    mut_prob: float = DEFAULT_GA_CONFIG.toolbox.mut_prob,
    tournament_size: int = DEFAULT_GA_CONFIG.toolbox.tournament_size,
    cv_folds: int = DEFAULT_GA_CONFIG.evaluator.cv_folds,
    n_estimators: Optional[int] = DEFAULT_GA_CONFIG.xgbrf.n_estimators,
    n_jobs: int = DEFAULT_GA_CONFIG.runner.n_jobs,
    early_stop_patience: Optional[int] = DEFAULT_GA_CONFIG.runner.early_stop_patience,
    early_stop_min_delta: float = DEFAULT_GA_CONFIG.runner.early_stop_min_delta,
    random_state: int = DEFAULT_GA_CONFIG.xgbrf.random_state,
    sep: str = DEFAULT_GA_CONFIG.ga_data.default_csv_sep,
) -> tuple[list[str], float]:
    if n_jobs <= 0:
        raise ValueError("n_jobs must be > 0")
    if early_stop_patience is not None and early_stop_patience <= 0:
        raise ValueError("early_stop_patience must be > 0 when provided")
    if early_stop_min_delta < 0.0:
        raise ValueError("early_stop_min_delta must be >= 0.0")

    X, y, feature_names = load_ga_input(
        csv_path=csv_path,
        label_column=label_column,
        sep=sep,
    )
    resolved_population_size = _resolve_population_size(
        requested_population_size=population_size,
        n_features=len(feature_names),
    )
    source_df = pd.read_csv(csv_path, sep=sep)

    evaluate_fn = build_xgbrf_evaluator(
        X=X,
        y=y,
        task=task,
        scoring=scoring,
        cv_folds=cv_folds,
        is_time_series=is_time_series,
        time_series_train_window=time_series_train_window,
        time_series_test_window=time_series_test_window,
        time_series_step_window=time_series_step_window,
        time_series_gap=time_series_gap,
        max_time_series_splits=max_time_series_splits,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    toolbox = build_toolbox(
        n_genes=len(feature_names),
        evaluate_fn=evaluate_fn,
        feature_names=feature_names,
        population_size=resolved_population_size,
        cx_prob=cx_prob,
        mut_prob=mut_prob,
        tournament_size=tournament_size,
    )

    pool: Optional[mp.pool.Pool] = None
    if n_jobs > 1:
        pool = mp.Pool(processes=n_jobs)
        toolbox.register("map", pool.map)

    try:
        random.seed(random_state)
        np.random.seed(random_state)
        population = toolbox.population(n=resolved_population_size)
        fitness_cache: dict[tuple[int, ...], float] = {}

        verbose = bool(DEFAULT_GA_CONFIG.runner.verbose)
        if verbose:
            print(
                "[GA] Starting run: "
                f"population={resolved_population_size}, generations={n_generations}, "
                f"cv_folds={cv_folds}, is_time_series={bool(is_time_series)}, "
                f"n_features={len(feature_names)}, n_jobs={n_jobs}",
                flush=True,
            )
            if population_size <= 0:
                print(
                    "[GA] Dynamic population sizing active: "
                    f"requested={population_size}, resolved={resolved_population_size}",
                    flush=True,
                )

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        _evaluate_invalid_individuals(
            toolbox,
            population,
            fitness_cache,
            verbose=verbose,
            phase_label="generation 0 initial population",
        )
        hof.update(population)
        gen0_stats = _population_stats(stats, population)
        best_so_far = _population_best_score(population)

        log_rows: list[dict[str, float | int]] = [
            {
                "gen": 0,
                "avg": gen0_stats["avg"],
                "std": gen0_stats["std"],
                "min": gen0_stats["min"],
                "max": gen0_stats["max"],
            }
        ]
        trace_rows: list[dict[str, float | int]] = [
            {
                "generation": 0,
                "best_going_in": best_so_far,
                "generation_best": best_so_far,
                "best_so_far": best_so_far,
                "avg": gen0_stats["avg"],
                "std": gen0_stats["std"],
                "min": gen0_stats["min"],
                "max": gen0_stats["max"],
            }
        ]

        if verbose:
            print(
                "Generation 0: "
                f"best_going_in={best_so_far:.6f} "
                f"generation_best={best_so_far:.6f} "
                f"best_so_far={best_so_far:.6f}",
                flush=True,
            )

        no_improvement_generations = 0
        for generation in range(1, n_generations + 1):
            best_going_in = best_so_far
            generation_started_at = time.perf_counter()
            offspring = algorithms.varAnd(
                population=population,
                toolbox=toolbox,
                cxpb=cx_prob,
                mutpb=mut_prob,
            )
            _evaluate_invalid_individuals(
                toolbox,
                offspring,
                fitness_cache,
                verbose=verbose,
                phase_label=f"generation {generation} offspring",
            )
            population = toolbox.select(offspring, k=resolved_population_size)
            hof.update(population)

            generation_stats = _population_stats(stats, population)
            generation_best = _population_best_score(population)
            best_so_far = max(best_so_far, generation_best)

            improvement = best_so_far - best_going_in
            if improvement > early_stop_min_delta:
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            log_rows.append(
                {
                    "gen": generation,
                    "avg": generation_stats["avg"],
                    "std": generation_stats["std"],
                    "min": generation_stats["min"],
                    "max": generation_stats["max"],
                }
            )
            trace_rows.append(
                {
                    "generation": generation,
                    "best_going_in": best_going_in,
                    "generation_best": generation_best,
                    "best_so_far": best_so_far,
                    "avg": generation_stats["avg"],
                    "std": generation_stats["std"],
                    "min": generation_stats["min"],
                    "max": generation_stats["max"],
                }
            )

            if verbose:
                generation_elapsed = time.perf_counter() - generation_started_at
                print(
                    f"Generation {generation}: "
                    f"best_going_in={best_going_in:.6f} "
                    f"generation_best={generation_best:.6f} "
                    f"best_so_far={best_so_far:.6f} "
                    f"elapsed={generation_elapsed:.1f}s",
                    flush=True,
                )

            if (
                early_stop_patience is not None
                and no_improvement_generations >= early_stop_patience
            ):
                if verbose:
                    print(
                        f"[GA] Early stopping triggered at generation {generation}: "
                        f"no improvement > {early_stop_min_delta} for "
                        f"{no_improvement_generations} generations.",
                        flush=True,
                    )
                break

        best_individual = hof[0]
        best_feature_names = decode_chromosome(list(best_individual), feature_names)
        best_score = float(best_individual.fitness.values[0])

        os.makedirs(output_dir, exist_ok=True)
        best_features_path = os.path.join(
            output_dir, DEFAULT_GA_CONFIG.runner.best_features_filename
        )
        best_features_csv_path = os.path.join(
            output_dir,
            DEFAULT_GA_CONFIG.runner.best_features_csv_filename,
        )
        best_feature_dataset_path = os.path.join(
            output_dir,
            DEFAULT_GA_CONFIG.runner.best_feature_dataset_filename,
        )
        generation_trace_path = os.path.join(
            output_dir,
            DEFAULT_GA_CONFIG.runner.generation_trace_filename,
        )
        logbook_path = os.path.join(output_dir, DEFAULT_GA_CONFIG.runner.logbook_filename)
        run_manifest_path = os.path.join(output_dir, DEFAULT_GA_CONFIG.runner.run_manifest_filename)

        with open(best_features_path, "w", encoding="utf-8") as f:
            for feature_name in best_feature_names:
                f.write(feature_name + "\n")
        pd.DataFrame({"feature_name": best_feature_names}).to_csv(
            best_features_csv_path, index=False
        )

        reduced_columns = [
            feature for feature in best_feature_names if feature in source_df.columns
        ]
        reduced_columns.append(label_column)
        source_df.loc[:, reduced_columns].to_csv(best_feature_dataset_path, index=False, sep=sep)

        logbook_df = pd.DataFrame(log_rows)
        logbook_df.to_csv(logbook_path, index=False)
        trace_df = pd.DataFrame(trace_rows)
        trace_df.to_csv(generation_trace_path, index=False)

        # Keep run manifest schema stable for backwards compatibility.
        run_manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "csv_path": os.path.abspath(csv_path),
            "label_column": label_column,
            "task": task,
            "scoring": scoring,
            "is_time_series": bool(is_time_series),
            "time_series_train_window": time_series_train_window,
            "time_series_test_window": time_series_test_window,
            "time_series_step_window": time_series_step_window,
            "time_series_gap": int(time_series_gap),
            "n_generations": int(n_generations),
            "population_size": int(resolved_population_size),
            "requested_population_size": int(population_size),
            "cx_prob": float(cx_prob),
            "mut_prob": float(mut_prob),
            "tournament_size": int(tournament_size),
            "cv_folds": int(cv_folds),
            "n_estimators": n_estimators,
            "random_state": int(random_state),
            "sep": sep,
            "best_score": best_score,
            "best_feature_count": len(best_feature_names),
            "output_dir": os.path.abspath(output_dir),
        }
        with open(run_manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(run_manifest, manifest_file, indent=2)

        print(f"Best features found: {len(best_feature_names)}", flush=True)
        print(f"Best score: {best_score:.4f}", flush=True)
        print(f"Output written to: {output_dir}", flush=True)

        return best_feature_names, best_score
    finally:
        if pool is not None:
            pool.close()
            pool.join()
