from collections import Counter
import json
import os
import sys

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import reduce_objective_name
from paramsweep_jobs import get_parameters, sample_params


colors = ["#509154", "#A9561E", "#77BCFD", "#B791D4", "#EEDD5D", 
          "#738696", "#24BCA8", "#D34A4F", "#8D81FE", "#FDA949"]
sns.set_palette(sns.color_palette(colors))


def save_data(network_size):
    properties_of_interest = json.load(open(f"objectives_{network_size}.json"))
    exp_dir = f"output/paramsweep/{network_size}"
    df = pd.DataFrame()
    for objectives in os.listdir(exp_dir):
        print(objectives)
        objective_path = f"{exp_dir}/{objectives}"
        if os.path.isfile(objective_path):
            continue
        for parami in os.listdir(objective_path):
            param_path = f"{objective_path}/{parami}"
            for replicate in os.listdir(param_path):
                rep_path = f"{param_path}/{replicate}"
                if os.path.isfile(rep_path) or replicate == "hpcc_out":
                    continue
                if len(os.listdir(rep_path)) == 0:
                    continue
                fitnesses = pd.read_pickle(f"{rep_path}/fitness_log.pkl")
                fitnesses = {k:v[-1] for k,v in fitnesses.items()}
                df_i = pd.read_csv(f"{rep_path}/diversity.csv")
                df_i = df_i.loc[df_i["property"].isin(properties_of_interest)]
                df_i["param_set"] = parami
                df_i["objectives"] = objectives
                df_i["rep"] = replicate
                for objective in fitnesses:
                    df_i.loc[df_i["property"] == objective, "optimized"] = "yes" if fitnesses[objective] == 0 else "no"
                df = pd.concat([df, df_i])
    df = df.reset_index()
    pd.to_pickle(df, f"{exp_dir}/df.pkl")


def keep_only_perfect_runs(df):
    key = ["objectives", "rep", "param_set"]
    perfect_runs = df.loc[df["optimized"] == "yes"].drop_duplicates(subset=key)[key]
    df = df.merge(perfect_runs, on=key, how="inner")
    df = df[df["optimized"].isna()]
    return df


def plot_parameter_performance(df, network_size, param_names, performance_metric):
    df = df.drop_duplicates(subset=["objectives", "rep", "param_set"])
    num_params = len(param_names)
    fig, ax = plt.subplots(1, num_params, figsize=(8*num_params,8))
    for p in range(num_params):
        sns.lineplot(data=df, x=param_names[p], y=performance_metric, hue="objectives", ax=ax[p])
    fig.tight_layout()
    plt.savefig(f"output/paramsweep/{network_size}/{performance_metric}.png")
    plt.close()


def plot_parameter_diversity(df, network_size, param_names, performance_metric):
    df = keep_only_perfect_runs(df)
    objectives = df["objectives"].unique()
    num_objectives = len(objectives)
    num_params = len(param_names)
    fig, ax = plt.subplots(num_objectives, num_params, figsize=(8*num_params,8*num_objectives))
    for o in range(num_objectives):
        df_o = df.loc[df["objectives"] == objectives[o]]
        for p in range(num_params):
            sns.lineplot(data=df_o, x=param_names[p], y=performance_metric, hue="property", ax=ax[o][p])
            ax[o][p].set_title(objectives[o])
    fig.tight_layout()
    plt.savefig(f"output/paramsweep/{network_size}/{performance_metric}.png")
    plt.close()


def score_params(df, param_names, performance_metric):
    df = keep_only_perfect_runs(df)
    key = ["objectives", "property", "rep"]
    df_grp = df.groupby(key)[performance_metric].idxmax()
    best_params = df.loc[df_grp]["param_set"].values
    best_counts = dict(Counter(best_params))
    df["best_count"] = df["param_set"].map(best_counts)
    df = df[param_names+["param_set", "best_count"]].drop_duplicates().dropna()
    print(df.sort_values("best_count", ascending=False))


def score_params_plot(df, network_size, performance_metric, filter=""):
    df = keep_only_perfect_runs(df)
    key = ["objectives", "property", "rep"]
    df_grp = df.groupby(key)[performance_metric].idxmax()
    best_params = df.loc[df_grp]

    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    sns.histplot(data=best_params, x="param_set_num", hue="objectives", multiple="stack", binwidth=1, ax=ax[0])
    sns.histplot(data=best_params, x="param_set_num", hue="property", multiple="stack", binwidth=1, ax=ax[1])
    fig.tight_layout()
    plt.savefig(f"output/paramsweep/{network_size}/{performance_metric}_best{filter}.png")
    plt.close()


def score_params_iter(df, network_size, param_names, performance_metric):
    score_params(df, param_names, performance_metric)
    score_params_plot(df, network_size, performance_metric)
    df = df.loc[df["popsize"] > 800]
    score_params(df, param_names, performance_metric)
    score_params_plot(df, network_size, performance_metric, "_popsize")
    df = df.loc[(df["age_gap"] > 300) & (df["age_gap"] < 500)]
    score_params(df, param_names, performance_metric)
    score_params_plot(df, network_size, performance_metric, "_popsize_agegap")


def main(network_size):
    try:
        df = pd.read_pickle(f"output/paramsweep/{network_size}/df.pkl")
    except:
        print("Please save the dataframe.")
        exit()
    
    params = get_parameters(int(network_size))
    param_names = list(params.keys())
    sampled_params = sample_params(100, param_names,
                                    [params[x]["low"] for x in params], [params[x]["high"] for x in params], 
                                    [params[x]["int"] for x in params], 42)
    df_params = pd.DataFrame(sampled_params)
    df_params["param_set"] = "params"+df_params.index.astype(str)
    df_params["param_set_num"] = df_params.index
    df = df_params.merge(df, on=["param_set"])
    df["perfect_pct"] = df["optimized_size"] / df["final_pop_size"]
    df = df.loc[df["property"] != "connectance"]

    plot_parameter_performance(df, network_size, param_names+["param_set_num"], "perfect_pct")
    plot_parameter_diversity(df, network_size, param_names+["param_set_num"], "spread")
    plot_parameter_diversity(df, network_size, param_names+["param_set_num"], "entropy")
    score_params_iter(df, network_size, param_names, "entropy")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        if sys.argv[2] == "save":
            save_data(sys.argv[1])
        else:
            print("Please provide a network size and \"save\"")
            print("if the parameter sweep dataframe has not yet been saved for the given network size.")
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide only a network size argument, if the dataframe has been saved.")