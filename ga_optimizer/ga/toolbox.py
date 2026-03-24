from __future__ import annotations

import random
from typing import Callable, List, Optional

from deap import base, creator, tools

from ga_optimizer.config import DEFAULT_GA_CONFIG, GAOptimizerConfig
from ga_optimizer.ga.data import decode_chromosome

try:
    creator.FitnessMax
except AttributeError:
    creator.create(
        "FitnessMax", base.Fitness, weights=DEFAULT_GA_CONFIG.toolbox.fitness_weights
    )

try:
    creator.Individual
except AttributeError:
    creator.create("Individual", list, fitness=creator.FitnessMax)


def build_toolbox(
    n_genes: int,
    evaluate_fn: Callable[[List[str]], float],
    feature_names: List[str],
    population_size: Optional[int] = None,
    cx_prob: Optional[float] = None,
    mut_prob: Optional[float] = None,
    tournament_size: Optional[int] = None,
    config: Optional[GAOptimizerConfig] = None,
) -> base.Toolbox:
    resolved_config = config or DEFAULT_GA_CONFIG
    resolved_population_size = (
        population_size
        if population_size is not None
        else resolved_config.toolbox.population_size
    )
    resolved_cx_prob = (
        cx_prob if cx_prob is not None else resolved_config.toolbox.cx_prob
    )
    resolved_mut_prob = (
        mut_prob if mut_prob is not None else resolved_config.toolbox.mut_prob
    )
    resolved_tournament_size = (
        tournament_size
        if tournament_size is not None
        else resolved_config.toolbox.tournament_size
    )

    toolbox = base.Toolbox()

    def evaluate_individual(individual: List[int]) -> tuple[float]:
        selected_features = decode_chromosome(list(individual), feature_names)
        score = evaluate_fn(selected_features)
        return (float(score),)

    toolbox.register(
        "attr_bool",
        random.randint,
        resolved_config.toolbox.attr_bool_min,
        resolved_config.toolbox.attr_bool_max,
    )
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_genes,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=resolved_mut_prob)
    toolbox.register("select", tools.selTournament, tournsize=resolved_tournament_size)

    # Keep these as explicit toolbox-level params for GA orchestration code.
    toolbox.population_size = resolved_population_size
    toolbox.cx_prob = resolved_cx_prob
    toolbox.mut_prob = resolved_mut_prob
    toolbox.tournament_size = resolved_tournament_size

    return toolbox
