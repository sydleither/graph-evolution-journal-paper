from collections import Counter
from itertools import combinations
import json
import os
from random import sample
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import reduce_objective_name
sys.path.insert(0, "./graph-evolution")
from bintools import numBins


output_dir = "output/sample_objectives"


def histograms(df, objectives, network_size):
    figure, axis = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    fig_row = 0
    fig_col = 0
    for objective in sorted(objectives):
        target = objectives[objective]
        if objective.endswith("distribution"):
            for dist in df[objective].values:
                axis[fig_row][fig_col].plot(dist, color="#509154")
            axis[fig_row][fig_col].plot(target, color="black")
        else:
            axis[fig_row][fig_col].hist(df[objective], bins=numBins(df[objective]), stacked=False, color="#509154")
            axis[fig_row][fig_col].axvline(target, color="black")
        axis[fig_row][fig_col].set_title(objective)
        fig_row += 1
        if fig_row % 2 == 0:
            fig_col += 1
            fig_row = 0
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure.savefig(f"output/sample_objectives/dist_{network_size}.png", bbox_inches="tight")
    plt.close()


def fixed_histograms(df, objectives, network_size, constraints):
    constraints_reduced = "".join([reduce_objective_name(x) for x in constraints])
    df_constrained = df
    for constraint in constraints:
        df_constrained = df_constrained.loc[df_constrained[constraint] == objectives[constraint]]
        if len(df_constrained) > 1:
            histograms(df_constrained, objectives, f"{network_size}_{constraints_reduced}")
        else:
            print(f"<=1 graphs with a {constraint} of {objectives[constraint]}.")


def entropy(df, objectives, network_size, popsize):
    entropies = {o:[] for o in objectives}
    for i in range(10):
        for objective in objectives:
            property_samples = sample(list(df[objective].values), popsize)
            type_counter = Counter(property_samples)
            entropy = -sum([(count/popsize)*np.log2(count/popsize) for count in type_counter.values()])
            entropies[objective].append(entropy)

    for o,e in entropies.items():
        print(o, np.mean(e), np.var(e), np.std(e))

    entropies_mean = {o:np.mean(e) for o,e in entropies.items()}
    with open(f"entropy_{network_size}.json", "w") as f:
        json.dump(entropies_mean, f, indent=4)


def seen_in_sample(df, objectives, network_size):
    num_samples = len(df)
    for num_objectives in range(1, 4):
        figure, axis = plt.subplots(figsize=(24,8))
        x = []
        y = []
        combos = list(combinations(objectives, num_objectives))
        for combo in sorted(combos):
            objective_names = "\n".join(reduce_objective_name(x) for x in combo)
            proportion_target = df
            for objective in combo:
                proportion_target = proportion_target.loc[proportion_target[objective] == 1]
            x.append(objective_names)
            y.append(len(proportion_target)/num_samples)
        axis.bar(x=x, height=y, color="#509154")
        axis.set(title="Proportion of Objective Values Seen in Random Sample")
        axis.set(xlabel="Objectives", ylabel="Proportion in Sample")
        figure.tight_layout()
        figure.savefig(f"output/sample_objectives/seen_{num_objectives}_{network_size}.png")


def main(network_size, limited):
    objectives = json.load(open(f"objectives_{network_size}.json"))
    if "connectance" not in objectives:
        objectives["connectance"] = 0.6

    if limited:
        output_name = f"{network_size}_limited.csv"
    else:
        output_name = f"{network_size}_full.csv"

    if not os.path.exists(f"{output_dir}/{output_name}"):
        print("Please run sample_objectives.py to generate the data to analyze.")
    else:
        df = pd.read_csv(f"{output_dir}/{output_name}")

    if limited:
        seen_in_sample(df, objectives, network_size)
    else:
        for objective in objectives:
            if objective.endswith("distribution"):
                df[objective] = df[objective].apply(lambda x: tuple([float(y) for y in x[1:-1].split(',')]))
        histograms(df, objectives, network_size)
        fixed_histograms(df, objectives, network_size, ["clustering_coefficient"])
        entropy(df, objectives, network_size, 10*network_size)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        try:
            network_size = int(sys.argv[2])
        except:
            print("Error converting network size argument into an integers.")
            exit()
        if sys.argv[1] == "full":
            main(network_size, limited=False)
        elif sys.argv[1] == "limited":
            main(network_size, limited=True)
        else:
            print("Please provide the sample type: full or limited.")
    else:
        print("Please read of the top of this file for usage instructions.")