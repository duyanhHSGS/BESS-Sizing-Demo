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
