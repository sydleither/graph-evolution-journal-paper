import json
import sys

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from common import get_network_sizes
from create_objectives import create_dist

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams.update({"figure.dpi": 300})


def reduce_objective_name(x):
    x = x.replace("interaction", "edge-weight").replace("_strength", "")
    x = "".join([y[0].upper() for y in x.split("_")])
    return x


def generate_dist_plots():
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    for i,network_size in enumerate(get_network_sizes()):
        dist = create_dist(network_size)
        ax[i].plot(list(range(network_size+1)), dist, linewidth=2, color="black")
        ax[i].set_title(f"Size {network_size} Network")
    fig.suptitle("Target Degree Distributions")
    fig.supxlabel("Degree")
    fig.supylabel("Proportion of Nodes")
    fig.tight_layout()
    plt.savefig("target_distributions.png")
    plt.close()


def histograms(df, objectives, network_size):
    figure, axis = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    fig_row = 0
    fig_col = 0
    for objective in sorted(objectives):
        target = objectives[objective]
        obj_name = reduce_objective_name(objective)
        if objective.endswith("distribution"):
            for dist in df[objective].values:
                axis[fig_row][fig_col].plot(dist, color="#f7879a")
            axis[fig_row][fig_col].plot(target, color="black")
        else:
            axis[fig_row][fig_col].hist(df[objective], bins=11, stacked=False, color="#f7879a")
            axis[fig_row][fig_col].axvline(target, color="black")
        axis[fig_row][fig_col].set_title(obj_name)
        fig_row += 1
        if fig_row % 2 == 0:
            fig_col += 1
            fig_row = 0
    figure.supxlabel("Property Value")
    figure.supylabel("Count")
    figure.suptitle("Distributions of Property Values from Randomly Sampling 5000 Graphs")
    figure.delaxes(axis[1][3])
    figure.tight_layout()
    figure.savefig(f"random_samples_{network_size}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    func_name = sys.argv[1]
    if func_name == "dd":
        generate_dist_plots()
    elif func_name == "sample":
        objectives = json.load(open(f"objectives_10.json"))
        if "connectance" not in objectives:
            objectives["connectance"] = 0.6
        df = pd.read_csv(f"output/sample_objectives/10_full.csv")
        for objective in objectives:
            if objective.endswith("distribution"):
                df[objective] = df[objective].apply(lambda x: tuple([float(y) for y in x[1:-1].split(',')]))
        histograms(df, objectives, 10)
    else:
        print("Invalid function given.")