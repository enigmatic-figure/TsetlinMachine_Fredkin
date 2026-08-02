#!/usr/bin/env python
"""Tests for the pure-Python Fredkin gate simulation."""

import unittest

from fredkin_logic import (
    FREDKIN_TRUTH_TABLE,
    fredkin_gate,
    fredkin_and,
    fredkin_not,
    fredkin_fanout,
    fredkin_literal_condition,
)


class FredkinGateTest(unittest.TestCase):
    def test_truth_table(self):
        expected_rows = (
            (0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 1),
            (0, 1, 0, 0, 1, 0),
            (0, 1, 1, 0, 1, 1),
            (1, 0, 0, 1, 0, 0),
            (1, 0, 1, 1, 1, 0),
            (1, 1, 0, 1, 0, 1),
            (1, 1, 1, 1, 1, 1),
        )
        self.assertEqual(FREDKIN_TRUTH_TABLE, expected_rows)

    def test_accepts_boolean_inputs(self):
        self.assertEqual(fredkin_gate(True, False, True), (1, 1, 0))

    def test_rejects_non_binary_inputs(self):
        with self.assertRaises(ValueError):
            fredkin_gate(2, 0, 1)

    def test_fredkin_universal_gates(self):
        self.assertEqual(fredkin_and(1, 1), 1)
        self.assertEqual(fredkin_and(1, 0), 0)
        self.assertEqual(fredkin_not(1), 0)
        self.assertEqual(fredkin_not(0), 1)
        self.assertEqual(fredkin_fanout(1), (1, 1))

    def test_fredkin_literal_condition(self):
        self.assertEqual(fredkin_literal_condition(action=1, literal_value=0), 0)
        self.assertEqual(fredkin_literal_condition(action=1, literal_value=1), 1)
        self.assertEqual(fredkin_literal_condition(action=0, literal_value=0), 1)


if __name__ == "__main__":
    unittest.main()
