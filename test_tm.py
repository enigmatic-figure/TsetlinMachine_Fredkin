# test_tm.py
import unittest
import numpy as np

from tsetlin_fredkin import MultiClassTsetlinMachine, FredkinTsetlinMachine
from control import WolframControl, ConstantControl


class TestTsetlinMachine(unittest.TestCase):
    def test_tm_can_learn_xor(self):
        X = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.int8,
        )

        y = np.array([0, 1, 1, 0], dtype=np.int32)

        tm = MultiClassTsetlinMachine(
            number_of_classes=2,
            number_of_clauses=8,
            number_of_features=2,
            number_of_states=50,
            s=5.0,
            threshold=5,
            seed=42,
        )

        X_big = np.repeat(X, 250, axis=0)
        y_big = np.repeat(y, 250)

        tm.fit(X_big, y_big, epochs=20)

        acc = tm.evaluate(X, y)
        self.assertGreaterEqual(acc, 0.99)

    def test_fredkin_tm_wolfram_control(self):
        X = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=np.int8,
        )

        y = np.array([0, 1, 1, 0], dtype=np.int32)

        tm = FredkinTsetlinMachine(
            number_of_classes=2,
            number_of_clauses=16,
            number_of_features=2,
            number_of_states=50,
            s=5.0,
            threshold=5,
            seed=1,
            control_program=WolframControl(rule_number=30),
            route_mode="features",
            edge_mode="xor",
            swap_when=1,
        )

        X_big = np.repeat(X, 250, axis=0)
        y_big = np.repeat(y, 250)

        tm.fit(X_big, y_big, epochs=20)

        acc = tm.evaluate(X, y)
        self.assertGreaterEqual(acc, 0.99)


if __name__ == "__main__":
    unittest.main()
