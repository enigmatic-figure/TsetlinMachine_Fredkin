# routing.py
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def cell_to_edge_controls(cell_controls, mode: str = "left") -> np.ndarray:
    """
    Convert per-cell controls into per-edge controls.

    If there are N cells, there are N-1 edges between cell i and i+1.

    Modes
    -----
    "left":
        edge[i] = cell[i]
    "right":
        edge[i] = cell[i+1]
    "xor":
        edge[i] = cell[i] XOR cell[i+1]
    "and":
        edge[i] = cell[i] AND cell[i+1]
    "or":
        edge[i] = cell[i] OR cell[i+1]
    """
    c = np.asarray(cell_controls, dtype=np.int8).ravel()
    n = c.size

    if n <= 1:
        return np.zeros(max(n - 1, 0), dtype=np.int8)

    if mode == "left":
        return c[:-1].astype(np.int8, copy=True)

    if mode == "right":
        return c[1:].astype(np.int8, copy=True)

    if mode == "xor":
        return (c[:-1] ^ c[1:]).astype(np.int8, copy=False)

    if mode == "and":
        return (c[:-1] & c[1:]).astype(np.int8, copy=False)

    if mode == "or":
        return (c[:-1] | c[1:]).astype(np.int8, copy=False)

    raise ValueError("mode must be 'left', 'right', 'xor', 'and', or 'or'")


class FredkinRouter:
    """
    Reversible 1D Fredkin message router.

    Messages have shape:
        (n_units, width)

    For width=1, 1D arrays of shape (n_units,) are also accepted.

    The router uses even/odd phases to avoid overlapping swaps.
    This makes the update reversible and easy to invert.
    """

    def __init__(
        self,
        n_units: int,
        width: int = 1,
        swap_when: int = 1,
        edge_mode: str = "left",
        phases: Sequence[int] = (0, 1),
    ):
        if n_units < 0:
            raise ValueError("n_units must be >= 0")

        if width < 1:
            raise ValueError("width must be >= 1")

        if swap_when not in (0, 1):
            raise ValueError("swap_when must be 0 or 1")

        self.n_units = int(n_units)
        self.width = int(width)
        self.swap_when = int(swap_when)
        self.edge_mode = edge_mode
        self.phases = tuple(phases)

        if any(phase not in (0, 1) for phase in self.phases):
            raise ValueError("phases currently support only 0 and 1")

    def _prepare_messages(self, messages, copy: bool = True) -> Tuple[np.ndarray, bool]:
        original_1d = np.asarray(messages).ndim == 1

        msgs = np.array(messages, dtype=np.int8, copy=copy)

        if msgs.ndim == 1:
            if self.width != 1:
                raise ValueError(
                    f"1D messages require width=1, got width={self.width}"
                )
            msgs = msgs.reshape(-1, 1)

        elif msgs.ndim == 2:
            if msgs.shape[1] != self.width:
                raise ValueError(
                    f"Expected messages with width={self.width}, got {msgs.shape[1]}"
                )

        else:
            raise ValueError("messages must be 1D or 2D")

        if msgs.shape[0] != self.n_units:
            raise ValueError(
                f"Expected {self.n_units} message rows, got {msgs.shape[0]}"
            )

        return msgs, original_1d

    def _prepare_edge_controls(
        self,
        cell_controls: Optional[np.ndarray],
        edge_controls: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.n_units <= 1:
            return np.zeros(0, dtype=np.int8)

        if edge_controls is not None:
            ec = np.asarray(edge_controls, dtype=np.int8).ravel()

            if ec.size != self.n_units - 1:
                raise ValueError(
                    f"Expected {self.n_units - 1} edge controls, got {ec.size}"
                )

            return ec

        if cell_controls is None:
            return np.zeros(self.n_units - 1, dtype=np.int8)

        cc = np.asarray(cell_controls, dtype=np.int8).ravel()

        if cc.size == self.n_units:
            return cell_to_edge_controls(cc, self.edge_mode)

        if cc.size == self.n_units - 1:
            return cc.astype(np.int8, copy=False)

        raise ValueError(
            "cell_controls must have length n_units or n_units-1, "
            f"got {cc.size} for n_units={self.n_units}"
        )

    def step(
        self,
        messages,
        cell_controls: Optional[np.ndarray] = None,
        edge_controls: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray]]]:
        """
        Apply one routed step.

        Returns
        -------
        routed_messages:
            Messages after Fredkin swaps.
        edge_controls:
            Edge controls actually used.
        history:
            History of swaps, usable by inverse().
        """
        msgs, original_1d = self._prepare_messages(messages, copy=True)
        ec = self._prepare_edge_controls(cell_controls, edge_controls)

        history: List[Tuple[int, np.ndarray]] = []

        if self.n_units < 2:
            if original_1d:
                return msgs.ravel(), ec, history
            return msgs, ec, history

        for phase in self.phases:
            idx = np.arange(phase, self.n_units - 1, 2, dtype=np.int64)

            if idx.size == 0:
                history.append((phase, np.empty(0, dtype=np.int64)))
                continue

            active = idx[ec[idx] == self.swap_when]

            if active.size > 0:
                left = msgs[active].copy()
                right = msgs[active + 1].copy()

                msgs[active] = right
                msgs[active + 1] = left

            history.append((phase, active.astype(np.int64, copy=False)))

        if original_1d:
            return msgs.ravel(), ec, history

        return msgs, ec, history

    def inverse(
        self,
        messages,
        history: List[Tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """
        Invert a previous step using its history.

        Because Fredkin swaps are self-inverse, applying the same swaps
        in reverse phase order restores the original messages.
        """
        msgs, original_1d = self._prepare_messages(messages, copy=True)

        for phase, active in reversed(history):
            if active.size == 0:
                continue

            left = msgs[active].copy()
            right = msgs[active + 1].copy()

            msgs[active] = right
            msgs[active + 1] = left

        if original_1d:
            return msgs.ravel()

        return msgs

    def permutation(
        self,
        cell_controls: Optional[np.ndarray] = None,
        edge_controls: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray]]]:
        """
        Return the permutation induced by the current controls.

        perm[new_index] = old_index

        This is useful if you want to route automaton actions or
        feature indices instead of raw feature bits.
        """
        idx = np.arange(self.n_units, dtype=np.int64)
        perm, ec, history = self.step(idx, cell_controls=cell_controls, edge_controls=edge_controls)
        return perm.astype(np.int64, copy=False), ec, history
