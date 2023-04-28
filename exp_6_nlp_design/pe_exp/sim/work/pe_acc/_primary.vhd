library verilog;
use verilog.vl_types.all;
entity pe_acc is
    port(
        mult_result     : in     vl_logic_vector(1023 downto 0);
        acc_result      : out    vl_logic_vector(31 downto 0)
    );
end pe_acc;
