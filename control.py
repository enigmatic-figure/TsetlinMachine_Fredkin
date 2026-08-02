# control.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


class ControlProgram:
    """
    Base class for control programs that produce CSWAP control bits.

    A control program receives a 1D binary state vector and returns
    a 1D binary control vector of the same length.

    These are cell controls. The router can convert cell controls
    into edge controls using several policies: left, right, xor, and, or.
    """

    def reset(self, n_units: int) -> None:
        """Optional hook for stateful controllers."""
        return None

    def control_bits(self, states: np.ndarray, t: int) -> np.ndarray:
        raise NotImplementedError


class ControlManager(ControlProgram):
    """
    Central manager for CSWAP control.

    This wraps another ControlProgram and allows manual overrides.
    This is useful for experimentation, debugging, and externally
    controlled switching routines.
    """

    def __init__(self, program: Optional[ControlProgram] = None):
        self.program = program
        self._override: Optional[np.ndarray] = None

    def set_override(self, bits) -> None:
        self._override = np.asarray(bits, dtype=np.int8)

    def clear_override(self) -> None:
        self._override = None

    def control_bits(self, states: np.ndarray, t: int) -> np.ndarray:
        states = _as_binary_states(states)

        if self._override is not None:
            bits = np.asarray(self._override, dtype=np.int8).ravel()

            if bits.size == states.size:
                return bits.astype(np.int8, copy=False)

            if bits.size == 1:
                return np.full(states.size, int(bits[0]), dtype=np.int8)

            raise ValueError(
                "Control override must be scalar or have the same length as states. "
                f"Got {bits.size}, expected {states.size}."
            )

        if self.program is None:
            return np.zeros(states.size, dtype=np.int8)

        return self.program.control_bits(states, t)


def _as_binary_states(states) -> np.ndarray:
    arr = np.asarray(states)
    if arr.ndim != 1:
        arr = arr.ravel()
    return (arr != 0).astype(np.int8)


def wolfram_table(rule_number: int) -> np.ndarray:
    """
    Return an 8-entry Wolfram elementary CA table.

    The neighborhood pattern is encoded as:
        pattern = 4*left + 2*center + right

    The output bit is:
        (rule_number >> pattern) & 1
    """
    if not (0 <= rule_number < 256):
        raise ValueError("Wolfram rule_number must be in [0, 255]")

    table = np.empty(8, dtype=np.int8)

    for pattern in range(8):
        table[pattern] = (rule_number >> pattern) & 1

    return table


def apply_wolfram_rule(
    states,
    rule_number: int,
    boundary: str = "wrap",
) -> np.ndarray:
    """
    Apply a Wolfram elementary CA rule to a 1D binary state vector.

    Parameters
    ----------
    states:
        1D binary array.
    rule_number:
        Wolfram rule number, 0..255.
    boundary:
        One of:
            "wrap"
            "zero"
            "reflect"
    """
    s = _as_binary_states(states)
    n = s.size

    if n == 0:
        return s.copy()

    if boundary == "wrap":
        left = np.roll(s, 1)
        right = np.roll(s, -1)

    elif boundary == "zero":
        left = np.concatenate(([0], s[:-1]))
        right = np.concatenate((s[1:], [0]))

    elif boundary == "reflect":
        left = np.concatenate(([s[0]], s[:-1]))
        right = np.concatenate((s[1:], [s[-1]]))

    else:
        raise ValueError("boundary must be 'wrap', 'zero', or 'reflect'")

    patterns = (left << 2) | (s << 1) | right
    table = wolfram_table(rule_number)

    return table[patterns].astype(np.int8, copy=False)


@dataclass
class WolframControl(ControlProgram):
    """
    Wolfram elementary CA control program.

    This is probably the most useful default controller for your experiment.
    """

    rule_number: int
    boundary: str = "wrap"
    invert: bool = False

    def control_bits(self, states: np.ndarray, t: int) -> np.ndarray:
        out = apply_wolfram_rule(states, self.rule_number, self.boundary)

        if self.invert:
            out = (out ^ 1).astype(np.int8, copy=False)

        return out


@dataclass
class ConstantControl(ControlProgram):
    """
    Constant control bits.

    value=0 with swap_when=1 means no swaps.
    value=1 with swap_when=1 means swap wherever edge conversion produces 1.
    """

    value: int = 0

    def control_bits(self, states: np.ndarray, t: int) -> np.ndarray:
        states = _as_binary_states(states)
        return np.full(states.size, int(self.value), dtype=np.int8)


@dataclass
class AlternatingControl(ControlProgram):
    """
    Alternating control bits.

    If spatial=False:
        all cells alternate in time.
    If spatial=True:
        cells alternate in space: 0,1,0,1,...
    """

    period: int = 2
    offset: int = 0
    spatial: bool = False

    def control_bits(self, states: np.ndarray, t: int) -> np.ndarray:
        states = _as_binary_states(states)

        if self.spatial:
            return ((np.arange(states.size) + self.offset) % 2).astype(np.int8)

        bit = ((t + self.offset) // max(1, self.period)) % 2
        return np.full(states.size, bit, dtype=np.int8)


@dataclass
class RandomControl(ControlProgram):
    """
    Random control bits.
    """

    probability: float = 0.5
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def control_bits(self, states: np.ndarray, t: int) -> np.ndarray:
        states = _as_binary_states(states)
        return (self._rng.random(states.size) < self.probability).astype(np.int8)


@dataclass
class SequenceControl(ControlProgram):
    """
    Sequence of scalar control values over time.

    Example:
        SequenceControl(values=np.array([0, 1, 1, 0]))
    """

    values: np.ndarray
    loop: bool = True

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.int8).ravel()

    def control_bits(self, states: np.ndarray, t: int) -> np.ndarray:
        states = _as_binary_states(states)

        if self.values.size == 0:
            return np.zeros(states.size, dtype=np.int8)

        if self.loop:
            idx = t % self.values.size
        else:
            idx = min(t, self.values.size - 1)

        return np.full(states.size, int(self.values[idx]), dtype=np.int8)


@dataclass
class CompositeControl(ControlProgram):
    """
    Combine multiple control programs with bitwise operations.

    op must be one of:
        "xor"
        "and"
        "or"
    """

    controls: Sequence[ControlProgram]
    op: str = "xor"

    def control_bits(self, states: np.ndarray, t: int) -> np.ndarray:
        states = _as_binary_states(states)

        if len(self.controls) == 0:
            return np.zeros(states.size, dtype=np.int8)

        acc = self.controls[0].control_bits(states, t).astype(np.int8, copy=True)

        for ctrl in self.controls[1:]:
            bits = ctrl.control_bits(states, t)

            if self.op == "xor":
                acc ^= bits
            elif self.op == "and":
                acc &= bits
            elif self.op == "or":
                acc |= bits
            else:
                raise ValueError("op must be 'xor', 'and', or 'or'")

        return acc.astype(np.int8, copy=False)
