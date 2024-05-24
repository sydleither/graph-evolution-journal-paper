import json

import numpy as np

from common import get_network_sizes


def create_exp_dist(network_size):
    #generate samples from exponential distribution
    ys = {10:1, 50:0.1, 100:0.05}
    basically_exp = [0]+[ys[network_size]*np.exp(-x*ys[network_size]) for x in range(1, network_size+1)]
    #make distribution sum up to 1
    sum_be = sum(basically_exp)
    basically_exp = [x/sum_be for x in basically_exp]
    #make distribution values achievable for the network size resolution
    basically_exp_r = [(1/network_size)*round(x*network_size) for x in basically_exp]
    #distribute rounding error across end of distribution
    round_diff = (1/network_size)*round((sum(basically_exp) - sum(basically_exp_r))*network_size)
    for i in range(1, network_size+1):
        if basically_exp_r[i] == 0 and round_diff > 0:
            j = 0
            round_diff_remainder = round_diff
            while round_diff_remainder > 0:
                basically_exp_r[i+j] = 1/network_size
                round_diff_remainder -= 1/network_size
                j += 1
            break
        elif basically_exp_r[i] == 0 and round_diff < 0:
            basically_exp_r[i-1] = basically_exp_r[i-1] + round_diff
            break
    #return a distribution where sum(p(k))=1 and the number of edges is an achievable integer
    return basically_exp_r


def main():
    for network_size in get_network_sizes():
        objectives = {
            "connectance": 0.2,
            "transitivity": 0.4,
            "positive_interactions_proportion": 0.75,
            "average_positive_interactions_strength": 0.75,
            "average_negative_interactions_strength": -0.25,
            "proportion_of_parasitism_pairs": 0.2,
            "in_degree_distribution": create_exp_dist(network_size),
            "out_degree_distribution": create_exp_dist(network_size)
        }

        with open(f"objectives_{network_size}.json", "w") as f:
            json.dump(objectives, f, indent=4)


if __name__ == "__main__":
    main()