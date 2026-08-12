# Brain HQ BESS Laboratory

Brain HQ is a Flask laboratory built around one canonical seven-eye `BrainEnv`.

- Brain 1 is a deterministic cheap/expensive tariff baseline.
- Brain 2 is a deterministic schedule/fill baseline.
- Brain 3 is the only trainable controller and uses a three-action DQN.
- Human BrainEnv is a manual physics and accounting playground.
- Oracle LP is an optional theoretical ceiling and never training input.

## Run

Use Python 3.10 and the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe main.py
```

Heavy Brain 3 training is intentionally performed on the user's other machine.

## Workflows

The dashboard contains seven workflows: Sizing, Brain 3 Training, Human BrainEnv,
Benchmarking, Live, Shadow, and Dispatch Viewer. Brain 1 and Brain 2 are built-in
controller choices inside comparison workflows; they do not have standalone tabs.

Dispatch, Benchmarking, Live, and Shadow all call the same measured-data runtime.
Every selected controller owns an independent BrainEnv while receiving identical
load, PV-derived net load, tariffs, and battery settings.

The dashboard restores the dense, flat OG visual language from `web/old.html` while
keeping only the seven Brain HQ workflows and current APIs. Sizing and Dispatch plot
one complete billing month at a time with OG bill strips, textual line toggles, hover
readouts, and explicit DEADLY peak overlays. Oracle and every Brain controller use
the same canonical period membership; Oracle is solved as one LP across the whole
month and Brain 3 runs one continuous BrainEnv episode across that same month.

Complete calendar months are preferred for billing episodes. If calendar data is
incomplete, the runtime warns and falls back to chronological sequential 30-day
periods; any trailing days that cannot complete a block are skipped with a warning.

Shadow never emits battery commands. It can replay local CSV data or completed
ThingsBoard telemetry and persists frozen results under `runs/shadow/`.

## Artifacts

Brain 3 deployment checkpoints are `checkpoints/brain3_<tag>.pt`. Full resume
state is stored separately as `checkpoints/brain3_resume_<tag>.pt`. Incompatible
schemas, sampling contracts, datasets, and sizing/economics fingerprints are
rejected rather than silently adapted.

See `git-plz-ignore/projarch.md` for the authoritative architecture map.
