import numpy as np


class ProbabilisticTsetlinAutomaton:
    """
    Probabilistic Tsetlin Automaton (PTM extension).

    Maintains a discrete probability distribution over TA states rather
    than a single deterministic integer state, enabling predictive uncertainty estimation.
    """

    def __init__(self, number_of_states: int):
        self.N = number_of_states
        self.state_prob = np.zeros(2 * number_of_states, dtype=np.float64)
        self.state_prob[number_of_states - 1] = 0.5
        self.state_prob[number_of_states] = 0.5

    def include_probability(self) -> float:
        return float(self.state_prob[self.N:].sum())

    def sample_action(self, rng=None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        state = rng.choice(2 * self.N, p=self.state_prob)
        return int(state >= self.N)
