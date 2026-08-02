#!/usr/bin/env python

def _as_bit(value, name):
    """Return *value* as a binary integer, rejecting non-bit inputs."""
    if value in (0, 1, False, True):
        return int(value)
    raise ValueError("%s must be a binary value (0/1 or bool), got %r" % (name, value))


def fredkin_gate(control, input_a, input_b):
    """Simulate a Fredkin gate with Python boolean logic.

    A Fredkin gate passes the control bit through unchanged. When control is 0,
    the two data inputs pass through unchanged; when control is 1, the two data
    inputs are swapped.

    Args:
        control: Control bit (0/1 or bool).
        input_a: First data bit (0/1 or bool).
        input_b: Second data bit (0/1 or bool).

    Returns:
        A tuple ``(control_out, output_a, output_b)`` of integer bits.
    """
    control_bit = _as_bit(control, "control")
    a_bit = _as_bit(input_a, "input_a")
    b_bit = _as_bit(input_b, "input_b")

    if control_bit:
        return control_bit, b_bit, a_bit
    return control_bit, a_bit, b_bit


FREDKIN_TRUTH_TABLE = tuple(
    (control, input_a, input_b) + fredkin_gate(control, input_a, input_b)
    for control in (0, 1)
    for input_a in (0, 1)
    for input_b in (0, 1)
)


if __name__ == "__main__":
    for row in FREDKIN_TRUTH_TABLE:
        print("c=%d a=%d b=%d -> c_out=%d a_out=%d b_out=%d" % row)
