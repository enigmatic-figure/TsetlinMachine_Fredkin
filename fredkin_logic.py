#!/usr/bin/env python
"""Pure-Python Fredkin (controlled-swap) gate simulation utilities.

This module provides universal Fredkin gate logic and logic synthesis helpers.
It supports both CSWAP (swap_when=1) and Toffoli/Fredkin paper (swap_when=0) conventions.
"""

from fredkin import (
    as_bit as _as_bit,
    fredkin_gate,
    fredkin_gate_array,
    fredkin_truth_table,
    fredkin_gate_paper,
    fredkin_and,
    fredkin_not,
    fredkin_fanout,
    fredkin_literal_condition,
)

FREDKIN_TRUTH_TABLE = fredkin_truth_table(swap_when=1)

if __name__ == "__main__":
    for row in FREDKIN_TRUTH_TABLE:
        print("c=%d a=%d b=%d -> c_out=%d a_out=%d b_out=%d" % row)
