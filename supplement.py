import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from common import get_network_sizes
from create_objectives import create_dist


def generate_dist_plots():
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    for i,network_size in enumerate(get_network_sizes()):
        dist = create_dist(network_size)
        ax[i].plot(list(range(network_size+1)), dist, linewidth=2, color="black")
        ax[i].set_title(f"Size {network_size} Network")
    fig.suptitle("Exponential Target Degree Distributions")
    fig.supxlabel("Degree")
    fig.supylabel("Proportion of Nodes")
    fig.tight_layout()
    plt.savefig("target_distributions.png")
    plt.close()


if __name__ == "__main__":
    func_name = sys.argv[1]
    if func_name == "dd":
        generate_dist_plots()
    else:
        print("Invalid function given.")