# experiments.py
import numpy as np

from control import WolframControl, RandomControl, ConstantControl
from tsetlin_fredkin import FredkinTsetlinMachine


def make_noisy_xor(n=5000, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)

    X = rng.integers(0, 2, size=(n, 2), dtype=np.int8)
    y = (X[:, 0] ^ X[:, 1]).astype(np.int32)

    noisy = rng.random(n) < noise
    y[noisy] ^= 1

    return X, y


def run_experiment(control_program, name):
    X, y = make_noisy_xor()

    tm = FredkinTsetlinMachine(
        number_of_classes=2,
        number_of_clauses=16,
        number_of_features=2,
        number_of_states=50,
        s=5.0,
        threshold=5,
        seed=1,
        control_program=control_program,
        route_mode="features",
        control_source="features",
        edge_mode="xor",
        swap_when=1,
    )

    tm.fit(X, y, epochs=50)
    acc = tm.evaluate(X, y)

    print(f"{name:25s} accuracy={acc:.4f}")
    return acc


if __name__ == "__main__":
    print("Running CSWAP Fredkin Tsetlin Machine Control Experiments:")
    print("-" * 55)
    run_experiment(ConstantControl(0), "constant_no_swap")
    run_experiment(ConstantControl(1), "constant_swap")
    run_experiment(RandomControl(probability=0.25, seed=1), "random_0.25")
    run_experiment(RandomControl(probability=0.50, seed=1), "random_0.50")

    for rule in [0, 30, 54, 60, 90, 110, 150, 255]:
        run_experiment(WolframControl(rule_number=rule), f"wolfram_rule_{rule}")
