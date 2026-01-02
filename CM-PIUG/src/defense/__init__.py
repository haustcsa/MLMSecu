# Defense module - Algorithm 2: Stackelberg-MFG defense
"""
Defense Module (Algorithm 2)
============================

Implements the Stackelberg Mean Field Game defense strategy optimization.

Key Components:
- MeanFieldGameSolver: Inner loop MFG equilibrium solver
- DefenseActionLibrary: Available defense actions (filter, rewrite, isolate, audit, constrain)
- StackelbergMFGSolver: Main solver combining offline SMFE and online matching

Two-Phase Process:
1. Offline Phase (SMFE Solving):
   - Inner loop: MFG equilibrium via best response + distribution consistency
   - Outer loop: Leader policy update via gradient estimation
   
2. Online Phase (Fast Matching):
   - Risk-based policy selection
   - Evidence-based adjustment
   - Action sampling from defense library

Convergence Parameters:
- ε_inner = 1e-4 (MFG convergence)
- ε_outer = 1e-3 (Stackelberg convergence)
"""

from .stackelberg_mfg import (
    StackelbergMFGSolver,
    MeanFieldGameSolver,
    DefenseActionLibrary,
    DefenseAction,
)

__all__ = [
    "StackelbergMFGSolver",
    "MeanFieldGameSolver",
    "DefenseActionLibrary",
    "DefenseAction",
]
