# Project structure

```text
main.py                       application entrypoint
bess/webapp.py                Flask composition root and JSON APIs
bess/brain/                   BrainEnv, Brain 1/2/3, sessions, shared runtime
bess/core/                    canonical settings, typed config, timebase helpers
bess/training/                Brain 3 launcher, runner, jobs, checkpoint discovery
bess/dispatch/                Brain dispatch orchestration and persisted runs
bess/evaluation/              sizing, Brain tournament, optional Oracle LP
bess/shadow/                  in-memory Live and persistent Shadow workflows
bess/integrations/            ThingsBoard telemetry connector
web/                          dashboard template, CSS, and JavaScript
data/                         source CSV datasets
tests/                        Brain-native static and runtime specifications
```

Generated state is intentionally narrow: Brain 3 artifacts live under `checkpoints/`,
Dispatch/Benchmark/Shadow histories live under `runs/`, and optional Oracle output
lives under `user_data/oracle_lp_cache/`. These stores use only the current BrainEnv
contract; raw CSV inputs and local ThingsBoard configuration are preserved separately.

Start the application from the repository root with `.\.venv\Scripts\python.exe main.py`.
Training is launched as `python -m bess.training.runners.train_brain3` by the web launcher.
No other trainable controller or environment is supported.
