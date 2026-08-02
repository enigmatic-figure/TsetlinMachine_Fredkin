# tsetlin_fredkin.py
from __future__ import annotations

from typing import Optional

import numpy as np

from control import (
    ControlManager,
    ControlProgram,
    ConstantControl,
    WolframControl,
)
from routing import FredkinRouter


class MultiClassTsetlinMachine:
    """
    A cleaned-up NumPy reference implementation of a multiclass Tsetlin Machine.

    This is intentionally explicit rather than maximally optimized.
    """

    def __init__(
        self,
        number_of_classes: int,
        number_of_clauses: int,
        number_of_features: int,
        number_of_states: int = 100,
        s: float = 10.0,
        threshold: int = 10,
        boost_true_positive_feedback: bool = False,
        clip_votes: bool = True,
        seed: Optional[int] = None,
    ):
        if number_of_classes <= 0:
            raise ValueError("number_of_classes must be > 0")

        if number_of_clauses <= 0:
            raise ValueError("number_of_clauses must be > 0")

        if number_of_features <= 0:
            raise ValueError("number_of_features must be > 0")

        if number_of_states <= 0:
            raise ValueError("number_of_states must be > 0")

        if s < 1.0:
            raise ValueError("s must be >= 1.0")

        if threshold <= 0:
            raise ValueError("threshold must be > 0")

        if number_of_clauses % number_of_classes != 0:
            raise ValueError(
                "number_of_clauses must be divisible by number_of_classes"
            )

        self.number_of_classes = int(number_of_classes)
        self.number_of_clauses = int(number_of_clauses)
        self.number_of_features = int(number_of_features)
        self.number_of_states = int(number_of_states)
        self.s = float(s)
        self.threshold = int(threshold)
        self.boost_true_positive_feedback = bool(boost_true_positive_feedback)
        self.clip_votes = bool(clip_votes)

        self.clauses_per_class = self.number_of_clauses // self.number_of_classes

        self.rng = np.random.default_rng(seed)

        # States are in [1, 2*N].
        # Initialize to N or N+1.
        # Action include iff state > N.
        self.ta_state = self.rng.integers(
            low=self.number_of_states,
            high=self.number_of_states + 2,
            size=(self.number_of_clauses, self.number_of_features, 2),
            dtype=np.int32,
        )

        # Clause indexing and polarity.
        self.clause_indices = np.arange(
            self.number_of_clauses,
            dtype=np.int32,
        ).reshape(self.number_of_classes, self.clauses_per_class)

        self.clause_sign = np.ones(
            (self.number_of_classes, self.clauses_per_class),
            dtype=np.int8,
        )
        self.clause_sign[:, 1::2] = -1

        self.clause_output = np.zeros(self.number_of_clauses, dtype=np.int8)
        self.class_sum = np.zeros(self.number_of_classes, dtype=np.float64)

        # Global learning/inference step counter.
        self.t = 0

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _preprocess_features(self, X, t: Optional[int] = None) -> np.ndarray:
        """
        Validate and binarize features.

        Subclasses can override this to route features through Fredkin gates.
        """
        X = np.asarray(X, dtype=np.int8).ravel()

        if X.size != self.number_of_features:
            raise ValueError(
                f"Expected {self.number_of_features} features, got {X.size}"
            )

        return (X != 0).astype(np.int8)

    def _action(self, state: np.ndarray) -> np.ndarray:
        return state > self.number_of_states

    def get_include_actions(self) -> np.ndarray:
        """
        Returns boolean array of shape:
            (number_of_clauses, number_of_features, 2)
        """
        return self.ta_state > self.number_of_states

    # ------------------------------------------------------------------
    # Clause evaluation
    # ------------------------------------------------------------------

    def calculate_clause_output(
        self,
        X: np.ndarray,
        predict: bool = False,
    ) -> np.ndarray:
        """
        Calculate clause outputs for a single input vector.

        Parameters
        ----------
        X:
            Binary feature vector of length number_of_features.
        predict:
            If True, all-excluded clauses output 0.
            If False, all-excluded clauses output 1, which is useful
            during training because the empty conjunction is true.
        """
        X_bool = np.asarray(X, dtype=bool)

        include = self.ta_state[:, :, 0] > self.number_of_states
        include_negated = self.ta_state[:, :, 1] > self.number_of_states

        # For each clause and feature:
        #   included positive literal requires X[k] == 1
        #   included negated literal requires X[k] == 0
        #
        # If a literal is excluded, it does not block the clause.
        ok_positive = (~include) | X_bool[None, :]
        ok_negated = (~include_negated) | (~X_bool[None, :])

        ok = ok_positive & ok_negated

        out = np.all(ok, axis=1).astype(np.int8)

        if predict:
            all_exclude = ~np.any(include | include_negated, axis=1)
            out[all_exclude] = 0

        self.clause_output = out
        return out

    def sum_up_class_votes(self) -> np.ndarray:
        votes = np.zeros(self.number_of_classes, dtype=np.float64)

        for target_class in range(self.number_of_classes):
            idx = self.clause_indices[target_class]
            sign = self.clause_sign[target_class]

            votes[target_class] = np.dot(self.clause_output[idx], sign)

            if self.clip_votes:
                votes[target_class] = np.clip(
                    votes[target_class],
                    -self.threshold,
                    self.threshold,
                )

        self.class_sum = votes
        return votes

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X, t: Optional[int] = None) -> int:
        X = self._preprocess_features(X, t=t)

        self.calculate_clause_output(X, predict=True)
        self.sum_up_class_votes()

        max_value = np.max(self.class_sum)
        candidates = np.where(self.class_sum == max_value)[0]

        if candidates.size == 1:
            return int(candidates[0])

        return int(self.rng.choice(candidates))

    def evaluate(self, X, y) -> float:
        X = np.asarray(X, dtype=np.int8)
        y = np.asarray(y, dtype=np.int32).ravel()

        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if X.shape[0] != y.size:
            raise ValueError("X and y have incompatible lengths")

        correct = 0

        for i in range(X.shape[0]):
            pred = self.predict(X[i])
            if pred == int(y[i]):
                correct += 1

        return float(correct) / float(max(1, X.shape[0]))

    # ------------------------------------------------------------------
    # Feedback selection
    # ------------------------------------------------------------------

    def _select_feedback(
        self,
        target_class: int,
        negative_target_class: int,
    ) -> np.ndarray:
        feedback = np.zeros(self.number_of_clauses, dtype=np.int8)

        T = float(self.threshold)

        # Target class feedback
        v_target = float(np.clip(self.class_sum[target_class], -T, T))
        p_target = (T - v_target) / (2.0 * T)

        if p_target > 0.0:
            draw = self.rng.random(self.clauses_per_class)
            selected = draw < p_target

            if np.any(selected):
                idx = self.clause_indices[target_class, selected]
                sign = self.clause_sign[target_class, selected]

                feedback[idx] = np.where(sign > 0, 1, -1).astype(np.int8)

        # Negative class feedback
        if negative_target_class >= 0:
            v_neg = float(np.clip(self.class_sum[negative_target_class], -T, T))
            p_neg = (T + v_neg) / (2.0 * T)

            if p_neg > 0.0:
                draw = self.rng.random(self.clauses_per_class)
                selected = draw < p_neg

                if np.any(selected):
                    idx = self.clause_indices[negative_target_class, selected]
                    sign = self.clause_sign[negative_target_class, selected]

                    feedback[idx] = np.where(sign > 0, -1, 1).astype(np.int8)

        return feedback

    # ------------------------------------------------------------------
    # Feedback application
    # ------------------------------------------------------------------

    def _apply_feedback(self, X: np.ndarray, feedback: np.ndarray) -> None:
        X_bool = np.asarray(X, dtype=bool)

        N = self.number_of_states
        max_state = 2 * N
        F = self.number_of_features

        penalty_prob = 1.0 / self.s
        reward_prob = 1.0 if self.boost_true_positive_feedback else (self.s - 1.0) / self.s

        for j in np.flatnonzero(feedback):
            fb = int(feedback[j])

            if fb > 0:
                # Type I feedback
                if self.clause_output[j] == 0:
                    draws = self.rng.random((F, 2))
                    dec = draws <= penalty_prob
                    self.ta_state[j] -= dec.astype(np.int32)

                else:
                    draw_inc = self.rng.random(F)

                    inc_reward = X_bool & (draw_inc <= reward_prob)
                    inc_penalty = (~X_bool) & (draw_inc <= penalty_prob)

                    if np.any(inc_reward):
                        self.ta_state[j, inc_reward, 0] += 1

                    if np.any(inc_penalty):
                        self.ta_state[j, inc_penalty, 0] -= 1

                    draw_neg = self.rng.random(F)

                    neg_reward = (~X_bool) & (draw_neg <= reward_prob)
                    neg_penalty = X_bool & (draw_neg <= penalty_prob)

                    if np.any(neg_reward):
                        self.ta_state[j, neg_reward, 1] += 1

                    if np.any(neg_penalty):
                        self.ta_state[j, neg_penalty, 1] -= 1

                np.clip(self.ta_state[j], 1, max_state, out=self.ta_state[j])

            elif fb < 0:
                # Type II feedback
                if self.clause_output[j] == 1:
                    include_action = self.ta_state[j, :, 0] > N
                    negated_action = self.ta_state[j, :, 1] > N

                    inc_mask = (~X_bool) & (~include_action)
                    neg_mask = X_bool & (~negated_action)

                    if np.any(inc_mask):
                        self.ta_state[j, inc_mask, 0] += 1

                    if np.any(neg_mask):
                        self.ta_state[j, neg_mask, 1] += 1

                    np.clip(self.ta_state[j], 1, max_state, out=self.ta_state[j])

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(self, X, target_class: int) -> None:
        target_class = int(target_class)

        if target_class < 0 or target_class >= self.number_of_classes:
            raise ValueError("target_class out of range")

        X = self._preprocess_features(X, t=self.t)

        self.calculate_clause_output(X, predict=False)
        self.sum_up_class_votes()

        if self.number_of_classes > 1:
            negative_target_class = int(self.rng.integers(0, self.number_of_classes - 1))
            if negative_target_class >= target_class:
                negative_target_class += 1
        else:
            negative_target_class = -1

        feedback = self._select_feedback(target_class, negative_target_class)
        self._apply_feedback(X, feedback)

        self.t += 1

    # ------------------------------------------------------------------
    # Batch training
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        epochs: int = 100,
        shuffle: bool = True,
    ) -> "MultiClassTsetlinMachine":
        X = np.asarray(X, dtype=np.int8)
        y = np.asarray(y, dtype=np.int32).ravel()

        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if X.shape[0] != y.size:
            raise ValueError("X and y have incompatible lengths")

        n = X.shape[0]

        for epoch in range(epochs):
            if shuffle:
                order = self.rng.permutation(n)
            else:
                order = np.arange(n)

            for example_id in order:
                self.update(X[example_id], int(y[example_id]))

        return self


class FredkinTsetlinMachine(MultiClassTsetlinMachine):
    """
    A Tsetlin Machine with a Fredkin-gate routing layer.

    The current implementation routes input features through a reversible
    1D Fredkin router before clause evaluation.
    """

    def __init__(
        self,
        number_of_classes: int,
        number_of_clauses: int,
        number_of_features: int,
        number_of_states: int = 100,
        s: float = 10.0,
        threshold: int = 10,
        boost_true_positive_feedback: bool = False,
        clip_votes: bool = True,
        seed: Optional[int] = None,
        control_program: Optional[ControlProgram] = None,
        router: Optional[FredkinRouter] = None,
        route_mode: str = "features",
        control_source: str = "features",
        edge_mode: str = "left",
        swap_when: int = 1,
    ):
        super().__init__(
            number_of_classes=number_of_classes,
            number_of_clauses=number_of_clauses,
            number_of_features=number_of_features,
            number_of_states=number_of_states,
            s=s,
            threshold=threshold,
            boost_true_positive_feedback=boost_true_positive_feedback,
            clip_votes=clip_votes,
            seed=seed,
        )

        if route_mode not in ("none", "features"):
            raise NotImplementedError(
                "Currently supported route_mode values: 'none', 'features'. "
            )

        self.route_mode = route_mode
        self.control_source = control_source

        self.control_manager = ControlManager(
            control_program if control_program is not None else ConstantControl(0)
        )

        if router is None:
            router = FredkinRouter(
                n_units=self.number_of_features,
                width=1,
                swap_when=swap_when,
                edge_mode=edge_mode,
            )

        if router.n_units != self.number_of_features:
            raise ValueError(
                "Router n_units must equal number_of_features. "
                f"Got router.n_units={router.n_units}, "
                f"number_of_features={self.number_of_features}."
            )

        self.router = router

        self.last_control: Optional[np.ndarray] = None
        self.last_edge_control: Optional[np.ndarray] = None
        self.last_history = None

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    def set_control_program(self, program: ControlProgram) -> None:
        self.control_manager.program = program

    def override_control(self, bits) -> None:
        self.control_manager.set_override(bits)

    def clear_control_override(self) -> None:
        self.control_manager.clear_override()

    def _control_states(self, X: np.ndarray) -> np.ndarray:
        F = self.number_of_features
        N = self.number_of_states

        if self.control_source == "features":
            return np.asarray(X, dtype=np.int8).ravel()

        if self.control_source == "include_actions":
            actions = self.ta_state[:, :, 0] > N
            density = np.mean(actions.astype(np.float64), axis=0)
            return (density >= 0.5).astype(np.int8)

        if self.control_source == "negated_actions":
            actions = self.ta_state[:, :, 1] > N
            density = np.mean(actions.astype(np.float64), axis=0)
            return (density >= 0.5).astype(np.int8)

        if self.control_source == "any_actions":
            include = self.ta_state[:, :, 0] > N
            negated = self.ta_state[:, :, 1] > N
            any_include = include | negated
            density = np.mean(any_include.astype(np.float64), axis=0)
            return (density >= 0.5).astype(np.int8)

        if self.control_source == "clause_outputs":
            if self.clause_output is None or self.clause_output.size == 0:
                return np.zeros(F, dtype=np.int8)

            return np.resize(self.clause_output.astype(np.int8), F)

        if self.control_source == "random":
            return self.rng.integers(0, 2, size=F, dtype=np.int8)

        raise ValueError(
            "control_source must be one of: "
            "'features', 'include_actions', 'negated_actions', "
            "'any_actions', 'clause_outputs', 'random'"
        )

    # ------------------------------------------------------------------
    # Feature routing
    # ------------------------------------------------------------------

    def _preprocess_features(self, X, t: Optional[int] = None) -> np.ndarray:
        X = super()._preprocess_features(X, t=t)

        if self.route_mode == "none":
            return X

        states = self._control_states(X)

        if t is None:
            t = self.t

        controls = self.control_manager.control_bits(states, t)

        routed, edge_controls, history = self.router.step(
            X,
            cell_controls=controls,
        )

        self.last_control = controls.astype(np.int8, copy=False)
        self.last_edge_control = edge_controls.astype(np.int8, copy=False)
        self.last_history = history

        return routed.astype(np.int8, copy=False)

    def route_actions_once(self, t: Optional[int] = None):
        dummy_X = np.zeros(self.number_of_features, dtype=np.int8)
        states = self._control_states(dummy_X)

        if t is None:
            t = self.t

        controls = self.control_manager.control_bits(states, t)

        perm, edge_controls, history = self.router.permutation(
            cell_controls=controls,
        )

        effective_ta_state = self.ta_state[:, perm, :]

        return effective_ta_state, perm, edge_controls, history


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    n = 4000
    X = rng.integers(0, 2, size=(n, 2), dtype=np.int8)

    y = (X[:, 0] ^ X[:, 1]).astype(np.int32)
    noise = rng.random(n) < 0.10
    y[noise] ^= 1

    baseline = FredkinTsetlinMachine(
        number_of_classes=2,
        number_of_clauses=16,
        number_of_features=2,
        number_of_states=50,
        s=5.0,
        threshold=5,
        seed=1,
        control_program=ConstantControl(0),
        route_mode="features",
        control_source="features",
        edge_mode="left",
        swap_when=1,
    )

    baseline.fit(X, y, epochs=50)
    print("Baseline accuracy:", baseline.evaluate(X, y))

    wolfram_tm = FredkinTsetlinMachine(
        number_of_classes=2,
        number_of_clauses=16,
        number_of_features=2,
        number_of_states=50,
        s=5.0,
        threshold=5,
        seed=1,
        control_program=WolframControl(rule_number=30, boundary="wrap"),
        route_mode="features",
        control_source="features",
        edge_mode="xor",
        swap_when=1,
    )

    wolfram_tm.fit(X, y, epochs=50)
    print("Wolfram-routed accuracy:", wolfram_tm.evaluate(X, y))
