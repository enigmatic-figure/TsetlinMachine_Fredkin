# test_routing.py
import unittest
import numpy as np

from routing import FredkinRouter, cell_to_edge_controls


class TestFredkinRouting(unittest.TestCase):
    def test_cell_to_edge_controls(self):
        cells = np.array([1, 0, 1, 1], dtype=np.int8)
        self.assertTrue(np.array_equal(cell_to_edge_controls(cells, "left"), [1, 0, 1]))
        self.assertTrue(np.array_equal(cell_to_edge_controls(cells, "right"), [0, 1, 1]))
        self.assertTrue(np.array_equal(cell_to_edge_controls(cells, "xor"), [1, 1, 0]))
        self.assertTrue(np.array_equal(cell_to_edge_controls(cells, "and"), [0, 0, 1]))
        self.assertTrue(np.array_equal(cell_to_edge_controls(cells, "or"), [1, 1, 1]))

    def test_router_inverse_random(self):
        rng = np.random.default_rng(123)

        router = FredkinRouter(
            n_units=10,
            width=1,
            swap_when=1,
            edge_mode="xor",
        )

        messages = rng.integers(0, 2, size=10, dtype=np.int8)
        controls = rng.integers(0, 2, size=10, dtype=np.int8)

        routed, edge_controls, history = router.step(
            messages,
            cell_controls=controls,
        )

        restored = router.inverse(routed, history)
        self.assertTrue(np.array_equal(messages, restored))

    def test_router_permutation(self):
        router = FredkinRouter(n_units=6, edge_mode="left", swap_when=1)
        controls = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        perm, ec, hist = router.permutation(cell_controls=controls)
        self.assertEqual(len(perm), 6)


if __name__ == "__main__":
    unittest.main()
