# fredkin.py
from __future__ import annotations

from typing import Tuple

import numpy as np


Bit = int


def as_bit(value, name: str) -> int:
    """
    Convert a value to a binary integer 0/1.

    Accepts bool, np.bool_, and integer 0/1 values.
    """
    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return int(value)

    raise ValueError(f"{name} must be a binary value (0/1 or bool), got {value!r}")


def fredkin_gate(
    control,
    input_a,
    input_b,
    *,
    swap_when: int = 1,
) -> Tuple[int, int, int]:
    """
    Scalar Fredkin gate / controlled SWAP.

    Parameters
    ----------
    control:
        Control bit.
    input_a:
        First data bit.
    input_b:
        Second data bit.
    swap_when:
        If control == swap_when, swap input_a and input_b.
        Use swap_when=1 for the common CSWAP convention.
        Use swap_when=0 to match the Fredkin/Toffoli paper convention.

    Returns
    -------
    (control_out, output_a, output_b)

    Notes
    -----
    The Fredkin gate paper uses:
        control = 0 -> swap
        control = 1 -> pass through

    Many engineering references use:
        control = 1 -> swap
        control = 0 -> pass through

    This function supports both.
    """
    if swap_when not in (0, 1):
        raise ValueError("swap_when must be 0 or 1")

    c = as_bit(control, "control")
    a = as_bit(input_a, "input_a")
    b = as_bit(input_b, "input_b")

    if c == swap_when:
        return c, b, a

    return c, a, b


def fredkin_gate_array(
    control,
    input_a,
    input_b,
    *,
    swap_when: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized Fredkin gate.

    All inputs are broadcast to a common shape.
    """
    if swap_when not in (0, 1):
        raise ValueError("swap_when must be 0 or 1")

    c = np.asarray(control, dtype=np.int8)
    a = np.asarray(input_a, dtype=np.int8)
    b = np.asarray(input_b, dtype=np.int8)

    c, a, b = np.broadcast_arrays(c, a, b)

    if not np.all((c == 0) | (c == 1)):
        raise ValueError("control array must contain only binary values")

    swap = c == swap_when

    out_a = np.where(swap, b, a).astype(np.int8, copy=False)
    out_b = np.where(swap, a, b).astype(np.int8, copy=False)

    return c.astype(np.int8, copy=False), out_a, out_b


def fredkin_truth_table(*, swap_when: int = 1) -> Tuple[Tuple[int, int, int, int, int, int], ...]:
    """
    Full truth table:
        control, a, b, control_out, a_out, b_out
    """
    return tuple(
        (c, a, b) + fredkin_gate(c, a, b, swap_when=swap_when)
        for c in (0, 1)
        for a in (0, 1)
        for b in (0, 1)
    )


# ----------------------------------------------------------------------
# Paper-convention helpers.
#
# The Fredkin/Toffoli paper convention:
#   control = 0 -> swap
#   control = 1 -> pass through
#
# With this convention:
#   AND(x1, x2):
#       set x3 = 0
#       y2 = x1 AND x2
#
#   NOT(x1):
#       set x2 = 1, x3 = 0
#       y3 = NOT x1
#
#   FANOUT(x1):
#       set x2 = 1, x3 = 0
#       y1 = x1, y2 = x1
#       y3 is garbage.
# ----------------------------------------------------------------------


def fredkin_gate_paper(control, input_a, input_b) -> Tuple[int, int, int]:
    """
    Fredkin gate using the convention from the Fredkin/Toffoli paper:
        control = 0 -> swap
        control = 1 -> pass through
    """
    return fredkin_gate(control, input_a, input_b, swap_when=0)


def fredkin_and(x1, x2) -> int:
    """
    Paper-convention Fredkin AND.

    Fix x3 = 0.
    Then y2 = x1 AND x2.
    """
    _, y2, _ = fredkin_gate_paper(x1, x2, 0)
    return y2


def fredkin_not(x1) -> int:
    """
    Paper-convention Fredkin NOT.

    Fix x2 = 1, x3 = 0.
    Then y3 = NOT x1.
    """
    _, _, y3 = fredkin_gate_paper(x1, 1, 0)
    return y3


def fredkin_fanout(x1) -> Tuple[int, int]:
    """
    Paper-convention Fredkin FANOUT.

    Fix x2 = 1, x3 = 0.
    Then y1 = x1 and y2 = x1.

    Returns
    -------
    (x1_copy_1, x1_copy_2)

    Notes
    -----
    In reversible logic, cloning is possible only because constants are supplied.
    There is still a garbage output, here y3 = NOT x1.
    """
    c, y1, _ = fredkin_gate_paper(x1, 1, 0)
    return c, y1


def fredkin_literal_condition(action: int, literal_value: int) -> int:
    """
    Fredkin-based include/exclude literal gate.

    Parameters
    ----------
    action:
        1 means include the literal.
        0 means exclude the literal.
    literal_value:
        The evaluated literal bit, e.g. x or NOT x.

    Returns
    -------
    1 if the literal condition is satisfied or if the literal is excluded.
    0 if the literal is included and the literal value is false.

    Implementation
    --------------
    Using paper convention:
        control = action
        input_a = literal_value
        input_b = 1

    If action = 1:
        no swap -> output_a = literal_value
    If action = 0:
        swap -> output_a = 1

    This gives:
        output = literal_value if include else 1
    """
    _, y2, _ = fredkin_gate_paper(action, literal_value, 1)
    return y2


if __name__ == "__main__":
    print("Fredkin truth table, swap_when=1")
    for row in fredkin_truth_table(swap_when=1):
        c, a, b, co, ao, bo = row
        print(f"c={c} a={a} b={b} -> c_out={co} a_out={ao} b_out={bo}")

    print()
    print("Fredkin truth table, paper convention swap_when=0")
    for row in fredkin_truth_table(swap_when=0):
        c, a, b, co, ao, bo = row
        print(f"c={c} a={a} b={b} -> c_out={co} a_out={ao} b_out={bo}")

    print()
    print("Fredkin AND using paper convention")
    for x1 in (0, 1):
        for x2 in (0, 1):
            print(f"AND({x1}, {x2}) = {fredkin_and(x1, x2)}")

    print()
    print("Fredkin NOT using paper convention")
    for x in (0, 1):
        print(f"NOT({x}) = {fredkin_not(x)}")

    print()
    print("Fredkin FANOUT using paper convention")
    for x in (0, 1):
        y1, y2 = fredkin_fanout(x)
        print(f"FANOUT({x}) = ({y1}, {y2})")
