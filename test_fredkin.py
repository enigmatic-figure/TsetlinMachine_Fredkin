# test_fredkin.py
import unittest
from itertools import product

from fredkin import (
    fredkin_gate,
    fredkin_and,
    fredkin_not,
    fredkin_fanout,
    fredkin_literal_condition,
)


class TestFredkinLogic(unittest.TestCase):
    def test_fredkin_self_inverse_swap_when_1(self):
        for c, a, b in product([0, 1], repeat=3):
            c1, a1, b1 = fredkin_gate(c, a, b, swap_when=1)
            c2, a2, b2 = fredkin_gate(c1, a1, b1, swap_when=1)
            self.assertEqual((c2, a2, b2), (c, a, b))

    def test_fredkin_self_inverse_swap_when_0(self):
        for c, a, b in product([0, 1], repeat=3):
            c1, a1, b1 = fredkin_gate(c, a, b, swap_when=0)
            c2, a2, b2 = fredkin_gate(c1, a1, b1, swap_when=0)
            self.assertEqual((c2, a2, b2), (c, a, b))

    def test_fredkin_conserves_multiset(self):
        for c, a, b in product([0, 1], repeat=3):
            c1, a1, b1 = fredkin_gate(c, a, b, swap_when=1)
            self.assertEqual(sorted([c, a, b]), sorted([c1, a1, b1]))

    def test_fredkin_universal_gates(self):
        # Test paper convention Fredkin AND
        self.assertEqual(fredkin_and(0, 0), 0)
        self.assertEqual(fredkin_and(0, 1), 0)
        self.assertEqual(fredkin_and(1, 0), 0)
        self.assertEqual(fredkin_and(1, 1), 1)

        # Test paper convention Fredkin NOT
        self.assertEqual(fredkin_not(0), 1)
        self.assertEqual(fredkin_not(1), 0)

        # Test paper convention Fredkin FANOUT
        self.assertEqual(fredkin_fanout(0), (0, 0))
        self.assertEqual(fredkin_fanout(1), (1, 1))

    def test_fredkin_literal_condition(self):
        # Excluded (action=0) -> output is 1
        self.assertEqual(fredkin_literal_condition(0, 0), 1)
        self.assertEqual(fredkin_literal_condition(0, 1), 1)

        # Included (action=1) -> output is literal value
        self.assertEqual(fredkin_literal_condition(1, 0), 0)
        self.assertEqual(fredkin_literal_condition(1, 1), 1)


if __name__ == "__main__":
    unittest.main()
