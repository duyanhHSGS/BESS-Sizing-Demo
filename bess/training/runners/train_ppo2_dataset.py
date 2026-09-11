"""Legacy CLI/import doorway for PPO2's root-owned runner.

The GUI/launcher now targets ``python -m ppo2.runner`` directly, while this
module remains runnable for existing scripts and tests.
"""
from ppo2.runner import *  # noqa: F401,F403
from ppo2.runner import _fit_test_split, _split_months, main


if __name__ == "__main__":
    main()
