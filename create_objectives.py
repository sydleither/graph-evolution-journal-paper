import json

from common import get_network_sizes


def create_dist(network_size):
    edges = {0.5:0.6, 0.6:0.1, 0.7:0.1, 0.8:0.1, 0.9:0.1}
    degrees = [x/network_size for x in range(network_size+1)]
    dd = [edges[x] if x in edges else 0 for x in degrees]
    return dd


def main():
    for network_size in get_network_sizes():
        objectives = {
            "clustering_coefficient": 0.7,
            "positive_interactions_proportion": 0.75,
            "average_positive_interactions_strength": 0.3,
            "variance_positive_interactions_strength": 0.08,
        }
        if network_size == 10:
            objectives["in_degree_distribution"] = create_dist(network_size)
            objectives["out_degree_distribution"] = create_dist(network_size)
        else:
            objectives["connectance"] = 0.6
        with open(f"objectives_{network_size}.json", "w") as f:
            json.dump(objectives, f, indent=4)


if __name__ == "__main__":
    main()