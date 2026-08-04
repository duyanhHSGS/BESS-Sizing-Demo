mini-faceIQ
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
process restarts.

Shadow Running
--------------

The **Shadow Running** tab evaluates No-BESS, SADRBC v13, and one local policy
against measured CSV days without sending battery commands. Save a source,
checkpoint, and battery body, then run catch-up for a date range. Configuration,
daily audit rows, and monthly virtual bills persist in `shadow/shadow.sqlite`
across Flask restarts. Reset the shadow history before changing its scientific
configuration.

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
`thingsboard_connector.py`, so the Shadow panel opens prefilled and can be
tested without re-entering the site fields.
