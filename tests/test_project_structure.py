from __future__ import annotations

import importlib
from pathlib import Path

from bess.agents import SUPPORTED_POLICY_ALGORITHMS
from bess.core.settings import PPO_TUNABLE_DEFAULTS
from bess.paths import PROJECT_ROOT
from bess.training.training_launcher import TRAINING_MODULES

OLD_ROOT_MODULES = {
    "common.py",
    "settings.py",
    "bess_env.py",
    "scenario_gen.py",
    "ppo_agent.py",
    "ppo2_agent.py",
    "training_launcher.py",
    "benchmark.py",
    "dispatch_runner.py",
    "thingsboard_connector.py",
}
REMOVED_POLICY_FILES = {
    PROJECT_ROOT / "bess" / "agents" / "grepo_agent.py",
    PROJECT_ROOT / "bess" / "agents" / "grepro_agent.py",
    PROJECT_ROOT / "bess" / "agents" / "pro_agent.py",
    PROJECT_ROOT / "bess" / "agents" / "sadrbc.py",
    PROJECT_ROOT / "bess" / "training" / "runners" / "train_grepo.py",
    PROJECT_ROOT / "bess" / "training" / "runners" / "train_grepro.py",
    PROJECT_ROOT / "bess" / "training" / "runners" / "train_pro.py",
    PROJECT_ROOT / "bess" / "forecasting" / "sadrbc_forecast.py",
}


def test_project_root_anchor_is_stable() -> None:
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert (PROJECT_ROOT / "data").is_dir()
    assert (PROJECT_ROOT / "bess").is_dir()


def test_old_root_modules_stay_moved() -> None:
    leftovers = sorted(name for name in OLD_ROOT_MODULES if (PROJECT_ROOT / name).exists())
    assert leftovers == []


def test_only_ppo_and_ppo2_are_supported() -> None:
    assert SUPPORTED_POLICY_ALGORITHMS == frozenset({"ppo", "ppo2"})
    assert set(TRAINING_MODULES) == {"ppo", "ppo2"}
    for module_name in TRAINING_MODULES.values():
        importlib.import_module(module_name)


def test_removed_policy_source_files_stay_deleted() -> None:
    assert [str(path.relative_to(PROJECT_ROOT)) for path in REMOVED_POLICY_FILES if path.exists()] == []


def test_flask_entrypoint_uses_moved_template_directory() -> None:
    main = importlib.import_module("main")
    assert main.app.template_folder == "web/templates"
    assert (PROJECT_ROOT / "web" / "templates" / "index.html").is_file()


def test_training_ui_wires_explicit_ppo2_fit_test_flag() -> None:
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="train-ppo2-fit-test"' in template
    assert 'ppo2_fit_test: document.getElementById("train-ppo2-fit-test").checked' in template
    assert "PPO2 FIT TEST:" in template


def test_dispatch_viewer_exposes_selected_ppo_eye6_with_distinct_style() -> None:
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "👁 Eye 6: running seen peak" in template
    assert 'policyKey: "ppo_eye6_running_peak_kw"' in template
    assert 'policy?.algo === "ppo"' in template
    assert 'const dispatchEye6Color = "#ff2bd6"' in template
    assert "series.eye6 ? 4 : 2" in template
    assert "Boolean(series.eye6)" in template
    assert "Number(Boolean(left.eye6)) - Number(Boolean(right.eye6))" in template
    assert 'if (checkbox.checked && selectedPolicy?.algo === "ppo")' in template
    assert "old broken Eye 6" in template
    assert "await createDispatchRunFor([policyName]);" in template
    assert "Exact PPO Eye 6 running-peak trace visible" in template


def test_dispatch_viewer_keeps_grid_display_meter_only() -> None:
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    block = template.split("function dispatchSeriesStyles() {", 1)[1].split("function policyDay(", 1)[0]

    assert '"load", "pv", "rolling_grid", "monthly_peak", "oracle_peak", "oracle_rolling_grid", "oracle_soc"' in block
    assert 'rolling_grid: "Daily 30-minute meter grid"' in block
    assert 'oracle_rolling_grid: "Oracle 30-minute meter grid"' in block
    assert 'label: `${label} 30-minute meter grid`' in block
    assert 'policyKey: "grid"' not in block
    assert 'policyKey: "discharge"' not in block
    assert "policyCharge: true" not in block
    assert "DISPATCH-METER-VIEW" in block


def test_training_ui_allows_ppo_full_dataset_fit_test() -> None:
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'item.style.display = "none";' in template
    assert "trainButton.disabled = days < 1 || !validControlDt;" in template
    assert "PPO FIT TEST:" in template
    assert "ALL ${days} supplied day(s) are reused for training, validation, and test" in template
    assert "trainButton.disabled = valDays < 1 || testDays < 1 || trainDays < 1" not in template


def test_generic_ppo_ui_exposes_and_sends_every_project_tunable() -> None:
    main = importlib.import_module("main")
    # Jinja parses the whole template here, catching broken {% ... %}/{{ ... }} syntax.
    main.app.jinja_env.get_template("index.html")
    template = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    field_by_setting = {
        "steps": "train-steps",
        "seed": "train-seed",
        "gamma": "train-gamma",
        "lambda": "train-lambda",
        "learning_rate": "train-learning-rate",
        "exploration_lr_multiplier": "train-exploration-lr-multiplier",
        "soc_edge_log_std_penalty": "train-soc-edge-log-std-penalty",
        "ppo_clip": "train-ppo-clip",
        "ppo_epochs": "train-ppo-epochs",
        "minibatch": "train-minibatch",
        "entropy_coef": "train-entropy-coef",
        "value_coef": "train-value-coef",
        "target_kl": "train-target-kl",
        "actor_grad_clip": "train-actor-grad-clip",
        "critic_grad_clip": "train-critic-grad-clip",
        "hidden_size": "train-hidden-size",
        "recurrent_enabled": "train-recurrent-enabled",
        "recurrent_sequence_length": "train-recurrent-sequence-length",
        "initial_log_std": "train-initial-log-std",
        "ppo_start_log_std": "train-ppo-start-log-std",
        "validate_every_updates": "train-validate-every-updates",
        "challenger_reset_patience": "train-challenger-reset-patience",
        "challenger_resets_enabled": "train-challenger-resets-enabled",
        "reset_optimizer_on_reanchor": "train-reset-optimizer-on-reanchor",
        "preserve_critic_on_reanchor": "train-preserve-critic-on-reanchor",
        "action_mismatch_shaping_scale": "train-action-mismatch-shaping-scale",
        "oracle_bc_enabled": "train-oracle-bc-enabled",
        "oracle_actor_bc_max_epochs": "train-oracle-actor-bc-max-epochs",
        "oracle_bc_max_epochs": "train-oracle-bc-max-epochs",
        "oracle_bc_learning_rate": "train-oracle-bc-learning-rate",
        "oracle_bc_minibatch": "train-oracle-bc-minibatch",
        "oracle_bc_target_mse": "train-oracle-bc-target-mse",
        "log_every_updates": "train-log-every-updates",
        "torch_threads": "train-torch-threads",
    }
    assert set(field_by_setting) == set(PPO_TUNABLE_DEFAULTS)
    field_ids = set(field_by_setting.values())
    for field_id in field_ids:
        assert f'id="{field_id}"' in template
        assert f'getElementById("{field_id}")' in template or field_id in {
            "train-steps",
            "train-gamma",
            "train-lambda",
        }
