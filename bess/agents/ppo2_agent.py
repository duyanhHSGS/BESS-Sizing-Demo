"""Compatibility import for PPO2's new root package.

New code should import :mod:`ppo2.agent` directly.
"""
from ppo2.agent import *  # noqa: F401,F403
from ppo2.agent import _adv_share_of_return  # noqa: F401
