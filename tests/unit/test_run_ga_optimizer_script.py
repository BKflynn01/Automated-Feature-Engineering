from pathlib import Path


def test_run_ga_optimizer_script_supports_config_and_preset_env():
    script = Path("run_ga_optimizer.sh").read_text(encoding="utf-8")
    assert "GA_CONFIG_PATH" in script
    assert "GA_PRESET" in script
    assert "--config" in script
    assert "--preset" in script


def test_run_ga_optimizer_script_removes_legacy_profile_support():
    script = Path("run_ga_optimizer.sh").read_text(encoding="utf-8")
    assert "GA_PROFILE" not in script
    assert 'GA_ARGS+=(--profile "${GA_PROFILE}")' not in script
