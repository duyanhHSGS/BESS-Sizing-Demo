"""Self-contained PPO2 controller family.

Application code should import the explicit PPO2 modules it needs (for example
``ppo2.agent`` or ``ppo2.env``). Keeping this package initializer lightweight
prevents settings-only callers from importing Torch and the full learner stack.
"""

# TODO(PPO2-HOME): keep this initializer lightweight; do not turn importing
# PPO2 settings into an eager import of training/inference machinery.
