`timescale 1ns/1ps

module fredkin_gate_tb;
    reg a_f;
    reg b_f;
    reg c_f;
    wire p_f;
    wire q_f;
    wire r_f;

    integer vector;
    integer failures;

    fredkin_gate dut (
        .a_f(a_f),
        .b_f(b_f),
        .c_f(c_f),
        .p_f(p_f),
        .q_f(q_f),
        .r_f(r_f)
    );

    initial begin
        failures = 0;

        for (vector = 0; vector < 8; vector = vector + 1) begin
            {a_f, b_f, c_f} = vector[2:0];
            #1;

            if (p_f !== a_f || q_f !== (a_f ? c_f : b_f) || r_f !== (a_f ? b_f : c_f)) begin
                $display(
                    "FAIL input=%b%b%b expected=%b%b%b actual=%b%b%b",
                    a_f, b_f, c_f,
                    a_f, (a_f ? c_f : b_f), (a_f ? b_f : c_f),
                    p_f, q_f, r_f
                );
                failures = failures + 1;
            end
        end

        if (failures == 0) begin
            $display("PASS fredkin_gate truth table");
        end else begin
            $display("FAIL fredkin_gate truth table failures=%0d", failures);
            $fatal(1);
        end
    end
endmodule
