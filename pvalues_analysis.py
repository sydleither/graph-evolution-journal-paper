import json
from math import ceil
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "./graph-evolution")
from bintools import numBins


def main(network_size):
    objectives = json.load(open("objectives.json"))
    df = pd.read_pickle(f"output/pvalues/df_{network_size}.pkl")
    #df = df.loc[(df["connectance"] > 0.2) & (df["connectance"] < 0.3)]
    num_samples = len(df)

    num_plots = len(df.columns)
    fig_col_cnt = 2 if num_plots <= 4 else 4
    fig_row_cnt = ceil(num_plots/fig_col_cnt)
    figure, axis = plt.subplots(fig_row_cnt, fig_col_cnt, figsize=(5*fig_row_cnt, 3*fig_col_cnt), squeeze=False)
    fig_row = 0
    fig_col = 0

    for objective in df.columns:
        target = objectives[objective]
        left_tail = len(df[df[objective] < target])
        right_tail = len(df[df[objective] > target])
        print(objective)
        print(f"\tLeft Tail: {left_tail/num_samples}")
        print(f"\tRight Tail: {right_tail/num_samples}")

        axis[fig_row][fig_col].hist(df[objective], bins=numBins(df[objective]), stacked=False, color="pink")
        axis[fig_row][fig_col].axvline(target, color="black")
        axis[fig_row][fig_col].set_title(objective)
        fig_row += 1
        if fig_row % fig_row_cnt == 0:
            fig_col += 1
            fig_row = 0

    figure.tight_layout()
    plt.savefig(f"output/pvalues/hist_{network_size}", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main(int(sys.argv[1]))