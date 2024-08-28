import json

from common import get_network_sizes


def create_dist(network_size):
    edges = {0.3:0.2, 0.4:0.6, 0.5: 0.2}
    degrees = [x/network_size for x in range(network_size+1)]
    dd = [edges[x] if x in edges else 0 for x in degrees]
    return dd


def main():
    for network_size in get_network_sizes():
        objectives = {
            "connectance": 0.4,
            "clustering_coefficient": 0.4,
            "positive_interactions_proportion": 0.75,
            "average_positive_interactions_strength": 0.3,
            "average_negative_interactions_strength": -0.3,
            "proportion_of_parasitism_pairs": 0.2,
            "in_degree_distribution": create_dist(network_size),
            "out_degree_distribution": create_dist(network_size)
        }

        with open(f"objectives_{network_size}.json", "w") as f:
            json.dump(objectives, f, indent=4)


if __name__ == "__main__":
    main()