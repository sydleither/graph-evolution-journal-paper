import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_palette(sns.color_palette("pastel"))

from common import reduce_objective_name


'''
Save Functions
'''
def create_row(objectives, rep, p, m, c, perfect, property, error=None, spread=None, uniformity=None):
    row = dict()
    row["objectives"] = objectives
    row["rep"] = rep
    row["popsize"] = p
    row["mutation_rate"] = m
    row["crossover"] = c
    row["perfect"] = perfect
    row["property"] = property
    row["error"] = error
    row["spread"] = spread
    row["uniformity"] = uniformity
    return row


def save_data(scheme_name, network_size):
    exp_dir = f"output/paramsweep/{scheme_name}"
    rows = []
    for objectives in os.listdir(exp_dir):
        obj_path = f"{exp_dir}/{objectives}"
        if os.path.isfile(obj_path):
            continue
        for param_combo in os.listdir(f"{obj_path}/{network_size}"):
            param_path = f"{obj_path}/{network_size}/{param_combo}"
            params = param_combo.split("_")
            p = params[0][1:]
            m = params[1][1:]
            c = params[2][1:]
            for replicate in os.listdir(param_path):
                rep_path = f"{param_path}/{replicate}"
                if os.path.isfile(rep_path) or replicate == "hpcc_out" or len(os.listdir(rep_path)) == 0:
                    continue
                rep = int(replicate)
                errores = pd.read_pickle(f"{rep_path}/fitness_log.pkl")
                errores = {k:v[-1] for k,v in errores.items()}
                perfect = len([x for x in errores.values() if x == 0]) == len(errores)
                for objective in errores.keys():
                    row = create_row(objectives, rep, p, m, c, perfect, objective, error=errores[objective])
                    rows.append(row)
                df_div = pd.read_csv(f"{rep_path}/diversity.csv")
                for property in ["clustering_coefficient", "positive_interactions_proportion"]:
                    df_prop = df_div.loc[df_div["property"] == property]
                    spread = df_prop["spread"].values[0]
                    uniformity = df_prop["uniformity"].values[0]
                    row = create_row(objectives, rep, p, m, c, perfect, 
                                     reduce_objective_name(property), 
                                     spread=spread, uniformity=uniformity)
                    rows.append(row)
    df = pd.DataFrame(rows)
    pd.to_pickle(df, f"{exp_dir}/{network_size}.pkl")
    return df


'''
Analysis Functions
'''
def plot_error(df, scheme_name, network_size):
    df = df.dropna(subset=["error"])
    figure, axis = plt.subplots(1, 3, figsize=(18, 6))
    for i,param_type in enumerate(["crossover", "mutation_rate", "popsize"]):
        sns.boxplot(data=df, x=param_type, y="error", hue="objectives", order=["low", "med", "high"], ax=axis[i])
    figure.tight_layout()
    plt.savefig(f"output/paramsweep/{scheme_name}/{network_size}_error.png")
    plt.close()


def count_perfect(df):
    df = df.dropna(subset=["error"])
    df_grouped = df[["crossover","mutation_rate","popsize","perfect"]].groupby(["crossover","mutation_rate","popsize"]).sum()
    print(df_grouped.sort_values(by="perfect", ascending=False))


def plot_diversity(df, scheme_name, network_size, diversity_measures):
    df = df.dropna(subset=diversity_measures)
    diversity_properties = df["property"].unique()
    for measure in diversity_measures:
        figure, axis = plt.subplots(2, 3, figsize=(18, 12))
        for j in range(len(diversity_properties)):
            for i,param_type in enumerate(["crossover", "mutation_rate", "popsize"]):
                sns.boxplot(data=df.loc[df["property"] == diversity_properties[j]], x=param_type,
                            y=measure, hue="objectives", order=["low", "med", "high"], ax=axis[j,i])
                axis[j,i].set(title=diversity_properties[j])
        figure.tight_layout()
        plt.savefig(f"output/paramsweep/{scheme_name}/{network_size}_{measure}.png")
        plt.close()


def average_diversity(df, diversity_measures):
    df = df.dropna(subset=diversity_measures)
    df_grouped = df[["crossover","mutation_rate","popsize"]+diversity_measures].groupby(["crossover","mutation_rate","popsize"]).mean()
    print(df_grouped.sort_values(by=diversity_measures, ascending=False))


def main(scheme_name, network_size):
    try:
        df = pd.read_pickle(f"output/paramsweep/{scheme_name}/{network_size}.pkl")
    except:
        print("Please save the dataframe.")
        exit()
    diversity_measures = ["spread", "uniformity"]
    
    plot_error(df, scheme_name, network_size)
    count_perfect(df)
    average_diversity(df, diversity_measures)
    plot_diversity(df, scheme_name, network_size, diversity_measures)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        if sys.argv[3] == "save":
            save_data(sys.argv[1], sys.argv[2])
        else:
            print("Invalid argument provided.")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Please provide a scheme name, network size, and optionally save.")