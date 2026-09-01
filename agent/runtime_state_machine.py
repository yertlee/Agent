"""Compatibility import for the M1 Supervisor without replacing legacy runtime.py."""
from .m1_runtime import CASConflict, InvalidTransition, RUN_TRANSITIONS, TASK_TRANSITIONS, StateMachine, Supervisor

__all__ = ["CASConflict", "InvalidTransition", "RUN_TRANSITIONS", "TASK_TRANSITIONS", "StateMachine", "Supervisor"]
