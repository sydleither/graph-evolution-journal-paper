import json
import sys


def main(network_size):
    objectives = json.load(open(f"objectives_{network_size}.json"))

    print("dd")
    dd = objectives["in_degree_distribution"]
    dd_edges = sum([i*network_size*dd[i] for i in range(1, network_size+1)])
    dd_c = dd_edges/network_size**2
    print(f"\tsum of pk: {sum(dd)}")
    print(f"\tnumber of edges: {dd_edges}")
    print(f"\tconnectance: ", dd_c)
    print(f"valid: {sum(dd) == 1 and dd_edges == int(dd_edges)}\n")

    print("c")
    c = dd_c
    max_c = network_size**2
    c_edges = c*max_c
    print(f"\tmax number of edges: {max_c}")
    print(f"\tnumber of edges with {c} connectance: {c_edges}")
    print(f"valid: {c_edges == int(c_edges)}\n")

    print("cpip")
    pip = objectives["positive_interactions_proportion"]
    cpip_edges = pip*c_edges
    print(f"\tnumber of positive edges with {pip} pip and {c} c edges: {cpip_edges}")
    print(f"valid: {cpip_edges == int(cpip_edges)}\n")


if __name__ == "__main__":
    main(int(sys.argv[1]))