"""
Fredkin-Tsetlin Machine Package

Integrates Conservative Logic Fredkin Universal Logic Gates and 1D Cellular Automata
(Wolfram CA Rules) into Tsetlin Machine Automata and Feature Routing.
"""

from fredkin import (
    fredkin_gate,
    fredkin_gate_array,
    fredkin_gate_paper,
    fredkin_and,
    fredkin_not,
    fredkin_fanout,
    fredkin_literal_condition,
)
from control import (
    ControlProgram,
    ControlManager,
    WolframControl,
    ConstantControl,
    AlternatingControl,
    RandomControl,
    SequenceControl,
    CompositeControl,
)
from routing import FredkinRouter, cell_to_edge_controls
from tsetlin_fredkin import MultiClassTsetlinMachine, FredkinTsetlinMachine
from Prob import ProbabilisticTsetlinAutomaton

__all__ = [
    "fredkin_gate",
    "fredkin_gate_array",
    "fredkin_gate_paper",
    "fredkin_and",
    "fredkin_not",
    "fredkin_fanout",
    "fredkin_literal_condition",
    "ControlProgram",
    "ControlManager",
    "WolframControl",
    "ConstantControl",
    "AlternatingControl",
    "RandomControl",
    "SequenceControl",
    "CompositeControl",
    "FredkinRouter",
    "cell_to_edge_controls",
    "MultiClassTsetlinMachine",
    "FredkinTsetlinMachine",
    "ProbabilisticTsetlinAutomaton",
]
