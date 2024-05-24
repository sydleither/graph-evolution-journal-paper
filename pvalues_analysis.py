import json
from itertools import combinations
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


def calculate_pvalue(df, objective, target):
    num_samples = len(df)
    if len(df) == 0:
        return 0
    if target > 0.5:
        return len(df[df[objective] > target])/num_samples
    else:
        return len(df[df[objective] < target])/num_samples


def significance_table(df, objectives):
    tables = dict()
    reduced = {x:reduce_objective_name(x) for x in objectives}
    for objective in objectives:
        objectives_minus_curr = [x for x in objectives if x != objective]
        for i in range(1, len(objectives_minus_curr)):
            combos = list(combinations(objectives_minus_curr, i))
            tables[i] = {objective:dict()}
            for combo in combos:
                df_combo = df
                col_name = ""
                for constraint in combo:
                    contstraint_target = objectives[constraint]
                    df_combo = df_combo.loc[(df_combo[constraint] > contstraint_target - 0.05) & 
                                            (df_combo[constraint] < contstraint_target + 0.05)]
                    col_name += reduced[constraint]
                pvalue = calculate_pvalue(df_combo, objective, objectives[objective])
                tables[i][objective][col_name] = round(pvalue, 5)
    return tables


def main(network_size):
    objectives = json.load(open(f"objectives_{network_size}.json"))
    df = pd.read_pickle(f"output/pvalues/df_{network_size}.pkl")

    histograms(df, objectives, network_size)
    del objectives["in_degree_distribution"]
    del objectives["out_degree_distribution"]
    significance_table(df, objectives)


if __name__ == "__main__":
    main(int(sys.argv[1]))