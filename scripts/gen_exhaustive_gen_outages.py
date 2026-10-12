"""Exhaustive single-generator-outage (N-1 gen) dataset for IEEE 118.

datakit's NMinusKGenerator enumerates branches only, so generator outages are
enumerated here by a drop-in TopologyGenerator and plugged in through the
`gen_n_minus_1` topology type. Everything else is the stock datakit pipeline.

    python scripts/gen_exhaustive_gen_outages.py configs/datakit/case118_exhaustive_n1_gen.yaml
"""

import sys
from typing import Generator

import gridfm_datakit.generate as gen_mod
from gridfm_datakit.network import Network
from gridfm_datakit.perturbations.topology_perturbation import TopologyGenerator
from gridfm_datakit.utils.idx_gen import GEN_BUS


class GenNMinus1Generator(TopologyGenerator):
    """Yield the N-0 base topology, then one topology per in-service non-reference generator."""

    def __init__(self, base_net: Network) -> None:
        super().__init__()
        self.gens = [
            int(i) for i in base_net.idx_gens_in_service if base_net.gens[i, GEN_BUS] != base_net.ref_bus_idx
        ]
        print(f"Number of possible topologies (N-0 + single gen outages): {1 + len(self.gens)}")

    def generate(self, net: Network) -> Generator[Network, None, None]:
        yield net.copy_for_perturbation()
        for g in self.gens:
            p = net.copy_for_perturbation()
            p.deactivate_gens([g])
            yield p


_stock = gen_mod.initialize_topology_generator


def _init(args, base_net):
    if args.type == "gen_n_minus_1":
        return GenNMinus1Generator(base_net)
    return _stock(args, base_net)


if __name__ == "__main__":
    gen_mod.initialize_topology_generator = _init
    gen_mod.generate_power_flow_data_distributed(sys.argv[1])
