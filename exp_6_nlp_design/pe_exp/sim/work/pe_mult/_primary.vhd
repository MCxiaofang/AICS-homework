library verilog;
use verilog.vl_types.all;
entity pe_mult is
    port(
        mult_neuron     : in     vl_logic_vector(511 downto 0);
        mult_weight     : in     vl_logic_vector(511 downto 0);
        mult_result     : out    vl_logic_vector(1023 downto 0)
    );
end pe_mult;
