import json
import sys


def main(network_size):
    objectives = json.load(open(f"objectives_{network_size}.json"))

    print("c")
    c = objectives["connectance"]
    max_c = network_size**2
    c_edges = c*max_c
    print(f"\tmax connectance: {max_c}")
    print(f"\tnumber of edges with {c} connectance: {c_edges}")
    print(f"valid: {c_edges == int(c_edges)}\n")

    print("popp")
    popp = objectives["proportion_of_parasitism_pairs"]
    total = (((network_size)*(network_size-1))/2)
    pairs = popp*total
    popp_edges = pairs*2
    print(f"\ttotal possible pairs: {total}")
    print(f"\t{popp} of total: {pairs}")
    print(f"\tnumber of edges: {popp_edges}")
    print(f"valid: {pairs == int(pairs)}\n")

    print("dd")
    dd = objectives["in_degree_distribution"]
    dd_edges = sum([i*network_size*dd[i] for i in range(1, network_size+1)])
    dd_c = dd_edges/network_size**2
    print(f"\tsum of pk: {sum(dd)}")
    print(f"\tnumber of edges: {dd_edges}")
    print(f"\tconnectance: ", dd_c)
    print(f"valid: {sum(dd) == 1 and dd_edges == int(dd_edges)}\n")

    print("cdd")
    print(f"\tc: {c}")
    print(f"\tdd connectance: {dd_c}")
    print(f"valid: {c == dd_c}\n")

    print("cpip")
    pip = objectives["positive_interactions_proportion"]
    cpip_edges = pip*c_edges
    print(f"\tnumber of positive edges with {pip} pip and {c} c edges: {cpip_edges}")
    print(f"valid: {cpip_edges == int(cpip_edges)}\n")

    print("cpopp")
    print(f"\tc edges: {c_edges}, popp edges: {popp_edges}")
    print(f"valid: {c_edges >= popp_edges}\n")

    print("cpippopp")
    print(f"\tcpip positive edges: {cpip_edges}, popp negative edges: {pairs}, c edges: {c_edges}")
    print(f"valid: {c_edges >= cpip_edges + pairs}\n")


if __name__ == "__main__":
    main(int(sys.argv[1]))