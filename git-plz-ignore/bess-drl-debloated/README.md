# bess-drl-debloated 🦍🔋

Same BESS DRL brain, dramatically less ceremony.

This folder keeps the existing **PPO training**, **observation contract**, **policy checkpoint contract**, **battery feasibility mapping**, **safety projection**, **MQTT telemetry accumulator**, **monthly demand state**, and **shadow/closed-loop dispatch behavior**.

It removes the service stack around them:

- no MongoDB
- no Beanie / Motor
- no repositories
- no FastAPI
- no REST CRUD layer
- no JobManager database records
- no APScheduler
- no httpx

Instead, everything important is explicit files + Python:

```text
policy_*.pt + MQTT
   (+ config.json only for old checkpoints)
              |
              v
         main.py run
              |
      observation -> PPO -> physics/safety
              |
       shadow: JSONL log
       closed: POST /api/plan
```

Training is equally boring (good boring):

```text
CSV + config.json
      |
      v
python main.py train ...
      |
existing run_train_dataset.py
      |
results/policy_TAG.pt
results/evaluation_TAG.json
results/training_curve_TAG.csv
```

## Why it reuses the existing core

`_core.py` adds `../bess-drl/src` to `sys.path` and imports only the pure DRL pieces. This is deliberate: there is one PPO/physics implementation, not a copied fork that silently drifts two weeks later 💀.

The debloated code never imports the original FastAPI app, Mongo models, repositories, controllers, or training JobManager.

If the workspace layout changes, set:

```bash
BESS_DRL_CORE_SRC=/path/to/bess-drl/src
```

## Install

Python 3.12+ recommended.

```bash
cd bess-drl-debloated
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The dependency list is intentionally small: NumPy, Pydantic, SciPy, PyTorch, gmqtt, and Flask. There is still no Mongo/Beanie/Motor/FastAPI stack.

## Web UI — Flask control room 💥

Start the local control panel:

```bash
python main.py ui
```

Then open:

```text
http://127.0.0.1:8080
```

Optional bind/port:

```bash
python main.py ui --host 0.0.0.0 --port 8080
```

The page is now the **literal original Sizing Demo `index.html`** supplied for the migration: the same inline CSS, HTML, JavaScript, tabs, labels, charts, buttons, and desktop layout. Flask only replaces the old backend underneath it.

```text
original Sizing Demo index.html
  -> Flask compatibility routes
     -> plain JSON/CSV/PT files
     -> existing PPO / BESSEnv / Oracle core
     -> python main.py train ... for training
     -> python main.py run ... for live MQTT control
```

The seven original tabs are kept: **Sizing Demo, Get Weather, Training Lab, Benchmarking, Live Runs, Shadow Running, and Dispatch Viewer**. The compatibility backend recreates their old browser APIs with local files instead of Mongo/FastAPI. Offline Dispatch/Live/Benchmark/Shadow simulation reuses the current `run_drl_policy` / `run_oracle` / scoring core; expensive replay outputs are fingerprint-cached under `state/`.

Training Lab forwards the advanced PPO controls into the current trainer, including validation/test month counts, PPO clip/epochs/minibatch, entropy/value coefficients, target KL, shaping margin, augmentation, behavior-cloning settings, and Torch CPU thread count. The current causal PPO contract is fixed at **15-minute control** and CPU execution; the UI reports an error rather than silently pretending unsupported custom control resolution or forced CUDA works.

Original Sizing form values persist in `state/ui_settings.json`. Generated Oracle/dispatch/live/shadow/weather/tournament files live under `state/` subfolders. The Flask development server is still intended as a local/operator UI, not an internet-facing auth gateway. No Mongo/Beanie/Motor/FastAPI stack was reintroduced.

## Train PPO directly

Every argument after `train` is passed untouched to the existing trainer, so all its hyperparameter flags still work.

```bash
python main.py train \
  --csv ../bess-drl/var/lib/bess-drl/datasets/YOUR_DATA.csv \
  --config-json config.example.json \
  --steps 1500000 \
  --tag my_policy \
  --seed 0
```

Or inspect the trainer's complete CLI:

```bash
python main.py train --help
```

Artifacts go straight to:

```text
bess-drl-debloated/results/
```

No training-job document. No policy registration. No database. The `.pt` file **is the policy**.

## Run in shadow mode

You need the same MQTT runtime telemetry topic used by the full service:

```text
bess-controller/runtime
```

Then:

```bash
python main.py run \
  --policy results/policy_my_policy.pt \
  --mode shadow \
  --mqtt-host localhost
```

Modern checkpoints already carry the exact `effective_config` used during training, and runtime uses that embedded snapshot. For an older checkpoint without it, add `--config config.example.json` as a fallback. If both exist, the checkpoint config wins so runtime cannot accidentally use different battery/tariff economics than training.

Decisions are appended to:

```text
logs/setpoints.jsonl
```

Monthly runtime state is persisted to:

```text
state/runtime_state.json
```

So restarting Python does not forget the running monthly demand peak / net-load history.

### Fast debug ticks

Production behavior is 15-minute ticks:

```bash
--interval 15 --offset 2
```

For development you can use:

```bash
python main.py run \
  --policy results/policy_my_policy.pt \
  --mode shadow \
  --interval 1
```

The observation still uses completed 15-minute billing/telemetry windows, matching the existing debug behavior.

## Closed loop

Closed mode pushes the same overlay payload shape to `POST /api/plan`:

```bash
python main.py run \
  --policy results/policy_my_policy.pt \
  --mode closed \
  --mqtt-host localhost \
  --controller-url http://localhost:8001 \
  --api-key dev-api-key-change-me
```

With no `--plan`, the base plan is 96 zero-kW `standby` slots, matching the full service's standalone fallback.

To overlay an existing planner plan, provide a JSON file:

```bash
--plan plan.json
```

Expected shape:

```json
{
  "date": "2026-08-14",
  "pPlan": [0.0],
  "dispatchSources": ["standby"],
  "socPlan": [],
  "socFloor": []
}
```

`pPlan` may be shorter and is padded to 96 by the existing overlay builder, but `dispatchSources` must contain exactly 96 entries because inventing the planner's dispatch intent would be wrong.

## Full-service thing -> debloated thing

| Full `bess-drl` | `bess-drl-debloated` |
|---|---|
| active policy in Mongo | `--policy policy_x.pt` |
| effective config in policy DB row | checkpoint `effective_config`; `--config` only fallback |
| `day_plans` Mongo collection | optional `--plan plan.json` |
| `drl_setpoint_logs` Mongo collection | `logs/setpoints.jsonl` |
| persistent JSON + Mongo metadata | persistent JSON only |
| APScheduler tick | asyncio sleep to aligned boundary |
| `httpx` controller POST | stdlib `urllib.request` |
| FastAPI training endpoint | `python main.py train ...` |
| JobManager subprocess | direct trainer subprocess |

## What is intentionally NOT reproduced

The HTTP management API, connector CRUD, Mongo policy registry, training job/SSE UI plumbing, and ThingsBoard fetch UI are service conveniences, not PPO behavior. They are intentionally absent.

If you already have a CSV, a config JSON, and a checkpoint, this folder is enough to train and run the DRL brain without Mongo doing interpretive dance in the background. 🦍💨
