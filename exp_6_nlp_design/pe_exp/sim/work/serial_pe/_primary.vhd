library verilog;
use verilog.vl_types.all;
entity serial_pe is
    port(
        clk             : in     vl_logic;
        rst_n           : in     vl_logic;
        neuron          : in     vl_logic_vector(15 downto 0);
        weight          : in     vl_logic_vector(15 downto 0);
        ctl             : in     vl_logic_vector(1 downto 0);
        vld_i           : in     vl_logic;
        result          : out    vl_logic_vector(31 downto 0);
        vld_o           : out    vl_logic
    );
end serial_pe;
