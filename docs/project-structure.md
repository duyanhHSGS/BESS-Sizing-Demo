# Project structure

The repository keeps only the Flask entrypoint at the Python root. Application code lives under the `bess` package and is grouped by responsibility.

```text
Sizing_Demo/
├── main.py
├── bess/
│   ├── paths.py
│   ├── core/
│   ├── agents/
│   ├── training/
│   │   └── runners/
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

- `bess/core`: shared configuration, BESS environment, scenario/data primitives.
- `bess/agents`: PPO, PPO2, GREPO, GrePRO, PRO, and SADRBC controllers.
- `bess/training`: datasets, launcher/jobs/checkpoints/reports, plus algorithm runners.
- `bess/evaluation`: baselines, benchmarking, benchmark storage/jobs, and Oracle LP/cache.
- `bess/dispatch`: checkpoint loading and dispatch execution/storage.
- `bess/forecasting`: weather forecasting and SADRBC/shadow forecast helpers.
- `bess/shadow`: live and shadow-run state/jobs.
- `bess/integrations`: external systems such as ThingsBoard.
- `benchmarks`: manual performance benchmarks; these are not unit tests.
- `scripts`: manually invoked maintenance/data-download utilities.
- `web/templates`: Flask templates.

## Commands

Start the web app from the repository root:

```bash
python main.py
```

Run tests:

```bash
python -m pytest
```

Training subprocesses are launched as package modules. Manual examples:

```bash
python -m bess.training.runners.train_ppo_dataset --help
python -m bess.training.runners.train_ppo2_dataset --help
python -m bess.training.runners.train_grepo --help
python -m bess.training.runners.train_grepro --help
python -m bess.training.runners.train_pro --help
```

`bess.paths.PROJECT_ROOT` is the canonical filesystem anchor. Modules must not assume their own directory is the repository root.
