import json
import sys

import matplotlib.pyplot as plt
import pandas as pd

from common import reduce_objective_name
sys.path.insert(0, "./graph-evolution")
from bintools import numBins


def histograms(df, objectives, network_size):
    figure, axis = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    fig_row = 0
    fig_col = 0
    for objective in sorted(objectives):
        target = objectives[objective]
        if objective.endswith("distribution"):
            for dist in df[objective].values:
                axis[fig_row][fig_col].plot(dist, color="pink")
            axis[fig_row][fig_col].plot(target, color="black")
        else:
            axis[fig_row][fig_col].hist(df[objective], bins=numBins(df[objective]), stacked=False, color="pink")
            axis[fig_row][fig_col].axvline(target, color="black")
        axis[fig_row][fig_col].set_title(objective)
        fig_row += 1
        if fig_row % 2 == 0:
            fig_col += 1
            fig_row = 0
    figure.tight_layout()
    plt.savefig(f"output/pvalues/hist_{network_size}", bbox_inches='tight')
    plt.close()


def truncate_values(df, objectives, num_dec=3):
    for objective in df.columns:
        if objective.endswith("distribution"):
            df[objective] = df[objective].apply(lambda dist: tuple([round(x, num_dec) for x in dist]))
            objectives[objective] = tuple([round(x, num_dec) for x in objectives[objective]])
        else:
            df[objective] = df[objective].apply(lambda x: round(x, num_dec))
            objectives[objective] = round(objectives[objective], num_dec)
    return df, objectives


def calculate_pvalue(df, objective, target):
    num_samples = len(df)
    if len(df) == 0:
        return 0
    #uniform distribution
    if objective.endswith("distribution") or objective in ["connectance", "clustering_coefficient"]:
        return len(df[df[objective] == target])/num_samples
    #normal distribution
    elif objective in ["positive_interactions_proportion", 
                       "average_positive_interactions_strength", 
                       "average_negative_interactions_strength"]:
        target = abs(target)
        if target > 0.5:
            return len(df[abs(df[objective]) >= target])/num_samples
        else:
            return len(df[abs(df[objective]) <= target])/num_samples
    #right-heavy distribution
    else:
        return len(df[df[objective] >= target])/num_samples


def pvalues(df, objectives, network_size):
    df, objectives = truncate_values(df, objectives)
    table = dict()
    for objective in objectives:
        contstraint_target = objectives[objective]
        pvalue = calculate_pvalue(df, objective, contstraint_target)
        table[objective] = round(pvalue, 5)
    with open(f"pvalues_{network_size}.json", "w") as f:
        json.dump(table, f, indent=4)


def fixed_histograms(df, objectives, network_size, constraints):
    constraints_reduced = "".join([reduce_objective_name(x) for x in constraints])
    df_constrained = df
    for constraint in constraints:
        df_constrained = df_constrained.loc[df_constrained[constraint] == objectives[constraint]]
    histograms(df_constrained, objectives, f"{network_size}_{constraints_reduced}")


def main(network_size):
    objectives = json.load(open(f"objectives_{network_size}.json"))
    df = pd.read_pickle(f"output/pvalues/df_{network_size}.pkl")

    histograms(df, objectives, network_size)
    pvalues(df, objectives, network_size)

    fixed_histograms(df, objectives, network_size, ["clustering_coefficient"])


if __name__ == "__main__":
    main(int(sys.argv[1]))