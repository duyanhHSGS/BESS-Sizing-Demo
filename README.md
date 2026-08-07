bESS
===========

How to run
----------

1. Use Python 3.10.

2. Create a virtual environment:

   python -m venv .venv

3. Activate the virtual environment:

   Windows PowerShell:
   .\.venv\Scripts\Activate.ps1

   macOS/Linux:
   source .venv/bin/activate

4. Install dependencies:

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

5. Run tests with pytest:

   python -m pytest

6. Start the app:

   python main.py

Live Runs
---------

Open the **Live Runs** tab, choose a local `policy_*.pt` checkpoint, and
create a session. A session snapshots the currently selected CSV and keeps
SADRBC, DRL, SOC, and peak state alive while you run one day at a time or use
Auto run. Sessions are held in server memory and disappear when the Flask
process restarts. The Live Runs day selector shows native-resolution Load, PV,
No-BESS grid, SADRBC grid, policy grid, and policy SOC traces in the same daily
shape as Dispatch Viewer. Click legend metrics to enable or disable individual
lines, use **Hide all lines** to clear the graph, and hover for time-slot values.

GrePRO hybrid controller
------------------------

New GrePRO checkpoints use SADRBC v13 as a causal baseline and learn only a
bounded, constant residual correction (5% by default). The training horizon
still grows from 3 to 7 to 30 days without simultaneously changing GrePRO's
physical authority. SADRBC receives
the portable real-weather forecast when forecast mode is selected. Otherwise
it receives a declared deterministic AR(1) forecast (default seed `130013`,
5% load error, 15% PV error). Exact future load/PV is never passed directly
to SADRBC. The seed and forecast contract are saved inside checkpoint meta so
Dispatch, Benchmarking, and Live Runs reproduce training behavior.

Shadow Running
--------------

The **Shadow Running** tab evaluates No-BESS, SADRBC v13, and one local policy
against measured CSV days without sending battery commands. Save a source,
checkpoint, and battery body, then run catch-up for a date range. Configuration,
daily audit rows, and monthly virtual bills persist in `shadow/shadow.sqlite`
across Flask restarts. Reset the shadow history before changing its scientific
configuration. Native-resolution daily traces are stored separately in the
same database and can be selected in the daily Shadow dispatch chart, with the
same metric toggles and hover inspector as Live Runs.

At the top of Shadow Running, the distinct **ThingsBoard Connector** panel can
configure and test the API URL, account, device ID, load/PV telemetry keys,
unit scaling, timezone, sampling interval, and maximum repaired gap. Selecting
ThingsBoard as the Shadow source fetches completed telemetry days directly,
caches valid raw days in the ignored `shadow/` folder, and reconstructs the
controller chain from that cache after restarts. Connector secrets remain in
the local ignored runtime folder and are never returned by the configuration
API.

The adjacent **Shadow Weather Forecast** panel configures Open-Meteo or a
compatible custom HTTPS provider with site coordinates, timezone, and optional
API-key authentication. For a ThingsBoard forecast policy, catch-up downloads
and caches real hourly weather, loads the checkpoint's saved causal ridge
`.npz`, reproduces the training feature pipeline, and generates the four live
forecast inputs without reading future load/PV actuals. Save Weather first,
then save the main Shadow configuration to freeze both data contracts.

The connector initially uses the Tande ThingsBoard `SPEC` embedded in
`bess/integrations/thingsboard_connector.py`, so the Shadow panel opens prefilled and can be
tested without re-entering the site fields.

Project layout
--------------

Application code lives under `bess/`, grouped by responsibility instead of being spread across the repository root. Manual utilities live in `scripts/`, performance probes live in `benchmarks/`, and Flask templates live in `web/templates/`. See `docs/project-structure.md` for the full map and package-module training commands.
