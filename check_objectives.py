import json
import sys

import pandas as pd

from common import reduce_objective_name


def main(network_size):
    objectives = json.load(open(f"objectives_{network_size}.json"))

    print("c")
    c = objectives["connectance"]
    max_c = network_size**2
    c_edges = c*max_c
    print(f"\tmax connectance: {max_c}")
    print(f"\tnumber of edges with {c} connectance: {c_edges}, valid: {c_edges == int(c_edges)}")

    print("popp")
    popp = objectives["proportion_of_parasitism_pairs"]
    total = (((network_size)*(network_size-1))/2)
    pairs = popp*total
    popp_edges = pairs*2
    print(f"\ttotal possible pairs: {total}")
    print(f"\t{popp} of total: {pairs}, valid: {pairs == int(pairs)}")
    print(f"\tnumber of edges: {popp_edges}")

    print("dd")
    dd = objectives["in_degree_distribution"]
    dd_edges = sum([i*network_size*dd[i] for i in range(1, network_size+1)])
    dd_c = dd_edges/network_size**2
    print(f"\tsum of pk: {sum(dd)}, valid: {sum(dd) == 1}")
    print(f"\tnumber of edges: {dd_edges}, valid: {dd_edges == int(dd_edges)}")
    print(f"\tconnectance: ", dd_c)

    print("cpip")
    pip = objectives["positive_interactions_proportion"]
    cpip_edges = pip*c_edges
    print(f"\tnumber of positive edges with {pip} pip and {c} c edges: {cpip_edges}, valid: {cpip_edges == int(cpip_edges)}")

    print("ddpip")
    ddpip_edges = pip*dd_edges
    print(f"\tnumber of positive edges with {pip} pip and {dd_c} dd edges: {ddpip_edges}, valid: {ddpip_edges == int(ddpip_edges)}")

    print("cpopp")
    print(f"\tc edges: {c_edges}, popp edges: {popp_edges}, valid: {c_edges >= popp_edges}")

    print("ddpopp")
    print(f"\tdd edges: {dd_edges}, popp edges: {popp_edges}, valid: {dd_edges >= popp_edges}")

    print("cpippopp")
    print(f"\tcpip positive edges: {cpip_edges}, popp negative edges: {pairs}, c edges: {c_edges}, valid: {c_edges >= cpip_edges + pairs}")

    print("ddpippopp")
    print(f"\tddpip positive edges: {ddpip_edges}, popp negative edges: {pairs}, dd edges: {dd_edges}, valid: {dd_edges >= ddpip_edges + pairs}")

    print("cc")
    cc = objectives["clustering_coefficient"]
    df = pd.read_pickle(f"output/pvalues/df_{network_size}.pkl")
    for objective in objectives:
        if objective == "clustering_coefficient":
            continue
        num_cc = len(df.loc[(df["clustering_coefficient"] == cc) & (df[objective] == objectives[objective])])
        print(f"\tnum random samples with cc and {reduce_objective_name(objective)}: {num_cc}, valid: {num_cc > 0}")


if __name__ == "__main__":
    main(int(sys.argv[1]))