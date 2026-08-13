# Project structure

The repository keeps only the Flask entrypoint at the Python root. Application code lives under the `bess` package and is grouped by responsibility.

```text
Sizing_Demo/
├── main.py
├── bess/
│   ├── paths.py
│   ├── core/
│   │   ├── config.py
│   │   ├── bess_env.py
│   │   ├── brain_runtime.py
│   │   └── ppo2_env.py
│   ├── agents/
│   │   ├── ppo_agent.py
│   │   └── ppo2_agent.py
│   ├── training/
│   │   └── runners/
│   │       ├── train_ppo_dataset.py
│   │       └── train_ppo2_dataset.py
│   ├── evaluation/
│   │   └── oracle/
│   ├── dispatch/
│   ├── forecasting/
│   ├── shadow/
│   └── integrations/
├── benchmarks/
├── scripts/
├── tests/
├── web/
│   └── templates/
├── data/
└── docs/
```

## Responsibilities

- `bess/core`: shared `BESSConfig`, billing/timebase helpers, the canonical seven-eye `BrainEnv`, PPO2's isolated senior-reference `PPO2Env`, and scenario/runtime primitives.
- `bess/agents`: only the supported learned controller families, PPO and PPO2. `bess.agents.SUPPORTED_POLICY_ALGORITHMS` is the shared allow-list.
- `bess/training`: dataset discovery/export, launch validation, jobs/checkpoints/reports, and the two PPO/PPO2 runners.
- `bess/evaluation`: No-BESS reference rollout, PPO/PPO2 rollout dispatch, checkpoint-centric tournament benchmarking, benchmark storage/jobs, and Oracle infrastructure.
- `bess/evaluation/oracle`: shared LP/cache reference plus PPO2's separate fixed-15-minute reference Oracle/scorer.
- `bess/dispatch`: PPO/PPO2 checkpoint loading, compatibility validation, dispatch execution, and persisted dispatch traces.
- `bess/forecasting`: standalone weather acquisition/forecast utilities; learned PPO/PPO2 policy observations do not receive forecast-only inputs.
- `bess/shadow`: in-memory Live Runs plus persisted No-BESS-vs-policy Shadow Running.
- `bess/integrations`: external systems such as ThingsBoard.
- `benchmarks`: manual performance probes; these are not unit tests.
- `scripts`: manually invoked maintenance/data-download utilities.
- `web/templates`: Flask UI.

## Controller rule

Only PPO and PPO2 are runnable learned controllers. No-BESS and Oracle are references, not controllers. Benchmarking is checkpoint-centric: compatible PPO/PPO2 `.pt` files compete as independent fighters, including multiple checkpoints from the same algorithm with different settings or seeds.

PPO uses canonical `BrainEnv`. PPO2 deliberately keeps its own fixed-15-minute playing field and must not be silently adapted into BrainEnv.

## Commands

Start the web app from the repository root:

```bash
python main.py
```

Run tests:

```bash
python -m pytest
```

Training subprocesses are launched as package modules:

```bash
python -m bess.training.runners.train_ppo_dataset --help
python -m bess.training.runners.train_ppo2_dataset --help
```

`bess.paths.PROJECT_ROOT` is the canonical filesystem anchor. Modules must not assume their own directory is the repository root.
