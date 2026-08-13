bESS
===========

How to run
----------

1. Use Python 3.10.
2. Create a virtual environment with `python -m venv .venv`.
3. Activate it (`.\.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux).
4. Install dependencies with `python -m pip install --upgrade pip` and `python -m pip install -r requirements.txt`.
5. Run tests with `python -m pytest`.
6. Start the app with `python main.py`.

Controller families
-------------------

The learned-controller surface intentionally contains only **PPO** and **PPO2**.

- **PPO** uses the canonical resolution-aware seven-eye `BrainEnv`.
- **PPO2** is the isolated senior-reference path. It keeps its dedicated 15-minute `PPO2Env`, 17-input contract, actor-only deployment wrapper, training runner, and private PPO2 Oracle/scorer.
- **No-BESS** and **Oracle LP** are references used for measurement; they are not learned controllers.

Training, Dispatch, Live Runs, Shadow Running, and Benchmarking accept only PPO/PPO2 checkpoints. Old checkpoint files from removed algorithms may still exist as runtime artifacts on another machine, but they are not runnable by this source tree.

Checkpoint Tournament
---------------------

The **Benchmarking** tab is checkpoint-centric. Select any compatible PPO/PPO2 `policy_*.pt` files and they fight as independent `.pt` contestants. This supports PPO-vs-PPO, PPO-vs-PPO2, different seeds, different training settings, and other checkpoint-level comparisons without giving any controller family a special benchmark seat.

No-BESS and the cached Oracle LP are neutral references. PPO2 participates only when the selected dataset satisfies its fixed 15-minute playing-field contract.

Live Runs
---------

Open **Live Runs**, choose a PPO/PPO2 `policy_*.pt`, and create a session. The selected CSV is snapshotted and the page reveals one day at a time beside the No-BESS reference. Daily traces contain Load, PV, No-BESS grid, policy grid, and policy SOC. Sessions live in server memory and disappear when the Flask process restarts.

Shadow Running
--------------

The **Shadow Running** tab evaluates No-BESS and one selected PPO/PPO2 checkpoint against measured CSV or ThingsBoard data without sending battery commands. Configuration, daily audit rows, monthly virtual bills, and native-resolution traces persist in `shadow/shadow.sqlite`.

The current source schema stores only No-BESS and policy results. If this checkout opens a legacy local Shadow database containing old controller-specific columns, that non-authoritative local history is reset and recreated under the current schema.

The **ThingsBoard Connector** panel configures and tests the API URL, account, device ID, load/PV telemetry keys, unit scaling, timezone, sampling interval, and repaired-gap limits. Connector secrets remain in ignored local runtime storage and are not returned by public configuration APIs.

Get Weather
-----------

The standalone **Get Weather** tab can download real hourly weather aligned to dated site data. This is independent tooling; PPO keeps its fixed seven-eye observation and PPO2 keeps its isolated senior-reference input contract.

Project layout
--------------

Application code lives under `bess/`, manual utilities under `scripts/`, performance probes under `benchmarks/`, Flask templates under `web/templates/`, and tests under `tests/`. See `docs/project-structure.md` and `git-plz-ignore/projarch.md` for the detailed architecture map.
