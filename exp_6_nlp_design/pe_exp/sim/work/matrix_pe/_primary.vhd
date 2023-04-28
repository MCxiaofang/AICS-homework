library verilog;
use verilog.vl_types.all;
entity matrix_pe is
    port(
        clk             : in     vl_logic;
        rst_n           : in     vl_logic;
        nram_mpe_neuron : in     vl_logic_vector(511 downto 0);
        nram_mpe_neuron_valid: in     vl_logic;
        nram_mpe_neuron_ready: out    vl_logic;
        wram_mpe_weight : in     vl_logic_vector(511 downto 0);
        wram_mpe_weight_valid: in     vl_logic;
        wram_mpe_weight_ready: out    vl_logic;
        ib_ctl_uop      : in     vl_logic_vector(7 downto 0);
        ib_ctl_uop_valid: in     vl_logic;
        ib_ctl_uop_ready: out    vl_logic;
        result          : out    vl_logic_vector(31 downto 0);
        vld_o           : out    vl_logic
    );
end matrix_pe;
