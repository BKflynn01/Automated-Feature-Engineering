import importlib
from dataclasses import replace

from ga_optimizer.config import DEFAULT_GA_CONFIG, ToolboxConfig
from ga_optimizer.ga.toolbox import build_toolbox


def test_build_toolbox_returns_configured_toolbox():
    feature_names = ["a", "b", "c", "d"]
    toolbox = build_toolbox(
        n_genes=4,
        evaluate_fn=lambda selected: float(len(selected)),
        feature_names=feature_names,
        population_size=50,
        cx_prob=0.5,
        mut_prob=0.2,
        tournament_size=3,
    )

    assert toolbox is not None
    assert hasattr(toolbox, "individual")
    assert hasattr(toolbox, "population")
    assert hasattr(toolbox, "evaluate")
    assert hasattr(toolbox, "mate")
    assert hasattr(toolbox, "mutate")
    assert hasattr(toolbox, "select")


def test_toolbox_evaluate_returns_length_one_tuple():
    feature_names = ["a", "b", "c", "d"]
    toolbox = build_toolbox(
        n_genes=4,
        evaluate_fn=lambda selected: float(len(selected)),
        feature_names=feature_names,
    )

    individual = toolbox.individual()
    individual[:] = [1, 0, 1, 0]
    result = toolbox.evaluate(individual)

    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0] == 2.0


def test_toolbox_population_builds_expected_shape():
    n_genes = 6
    toolbox = build_toolbox(
        n_genes=n_genes,
        evaluate_fn=lambda selected: float(len(selected)),
        feature_names=[f"f{i}" for i in range(n_genes)],
    )

    population = toolbox.population(n=10)
    assert isinstance(population, list)
    assert len(population) == 10
    assert all(len(individual) == n_genes for individual in population)


def test_toolbox_defaults_follow_shared_config():
    toolbox = build_toolbox(
        n_genes=4,
        evaluate_fn=lambda selected: float(len(selected)),
        feature_names=["a", "b", "c", "d"],
    )

    assert toolbox.population_size == DEFAULT_GA_CONFIG.toolbox.population_size
    assert toolbox.cx_prob == DEFAULT_GA_CONFIG.toolbox.cx_prob
    assert toolbox.mut_prob == DEFAULT_GA_CONFIG.toolbox.mut_prob
    assert toolbox.tournament_size == DEFAULT_GA_CONFIG.toolbox.tournament_size


def test_toolbox_uses_custom_toolbox_config_values():
    config = replace(
        DEFAULT_GA_CONFIG,
        toolbox=ToolboxConfig(
            fitness_weights=(1.0,),
            attr_bool_min=0,
            attr_bool_max=0,
            population_size=12,
            cx_prob=0.6,
            mut_prob=0.15,
            tournament_size=4,
        ),
    )
    toolbox = build_toolbox(
        n_genes=5,
        evaluate_fn=lambda selected: float(len(selected)),
        feature_names=["a", "b", "c", "d", "e"],
        config=config,
    )
    individual = toolbox.individual()

    assert toolbox.population_size == 12
    assert toolbox.cx_prob == 0.6
    assert toolbox.mut_prob == 0.15
    assert toolbox.tournament_size == 4
    assert individual == [0, 0, 0, 0, 0]


def test_toolbox_module_is_safe_to_reload_with_deap_creator_types_registered():
    import ga_optimizer.ga.toolbox as toolbox_module

    importlib.reload(toolbox_module)
