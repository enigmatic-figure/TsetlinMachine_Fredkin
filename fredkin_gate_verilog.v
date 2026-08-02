`timescale 1ns/1ps

// Fredkin (controlled-swap) gate.
//
// When the control input (a_f) is low, b_f and c_f pass through unchanged.
// When a_f is high, b_f and c_f are swapped. The control bit is always
// forwarded to p_f so the operation remains reversible.
module fredkin_gate(
    input  wire a_f,
    input  wire b_f,
    input  wire c_f,
    output wire p_f,
    output wire q_f,
    output wire r_f
);

assign p_f = a_f;
assign q_f = a_f ? c_f : b_f;
assign r_f = a_f ? b_f : c_f;

endmodule
