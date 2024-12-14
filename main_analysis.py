import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from main_jobs import get_diversity_funcs, get_network_sizes


colors = ["#f7879a", "#509154", "#A9561E", "#77BCFD", "#B791D4", 
          "#EEDD5D", "#738696", "#24BCA8", "#D34A4F", "#8D81FE"]
sns.set_palette(sns.color_palette(colors))

edge_weight_properties = ["positive_interactions_proportion",
                          "average_positive_interactions_strength",
                          "variance_positive_interactions_strength"]
topological_properties = ["connectance",
                          "clustering_coefficient",
                          "in_degree_distribution",
                          "out_degree_distribution"]


def reduce_objective_names(obj_names):
    obj_names = sorted(obj_names)
    obj_names = ["".join([x[0].upper() for x in y.split("_")]) for y in obj_names]
    return "\n".join(obj_names)


def save_data(network_size):
    all_properties = json.load(open(f"objectives_{network_size}.json"))
    exp_dir = f"output/main/{network_size}"
    df = pd.DataFrame()
    for num_objectives in os.listdir(exp_dir):
        objective_path = f"{exp_dir}/{num_objectives}"
        if os.path.isfile(objective_path):
            continue
        print(f"Saving {num_objectives}...")
        for num_exp in os.listdir(objective_path):
            exp_path = f"{objective_path}/{num_exp}"
            if os.path.isfile(exp_path):
                continue
            for replicate in os.listdir(exp_path):
                rep_path = f"{exp_path}/{replicate}"
                if os.path.isfile(rep_path) or replicate == "hpcc_out":
                    continue
                if len(os.listdir(rep_path)) == 0:
                    print(f"missing: output/main/{network_size}/{num_objectives}/{num_exp}/{replicate}")
                    continue
                fitnesses = pd.read_pickle(f"{rep_path}/fitness_log.pkl")
                fitnesses = {k:v[-1] for k,v in fitnesses.items()}
                objective_properties = list(fitnesses.keys())
                div_funcs = get_diversity_funcs(network_size, all_properties, objective_properties)
                properties_of_interest = div_funcs + objective_properties
                df_i = pd.read_csv(f"{rep_path}/diversity.csv")
                df_i = df_i.loc[df_i["property"].isin(properties_of_interest)]
                objectives = reduce_objective_names(objective_properties)
                df_i["objectives"] = objectives
                df_i["exp_num"] = num_exp
                df_i["num_objectives"] = num_objectives
                df_i["rep"] = replicate
                df_i["objective"] = False
                df_i.loc[df_i["property"].isin(objective_properties), "objective"] = True
                df = pd.concat([df, df_i])
    df = df.reset_index()
    pd.to_pickle(df, f"{exp_dir}/df.pkl")


def keep_only_perfect_runs(df):
    avg_performance = df[["uid", "optimized_proportion"]].groupby("uid").mean().reset_index()
    perfect_runs = avg_performance.loc[avg_performance["optimized_proportion"] == 1]["uid"]
    df = df.merge(perfect_runs, on="uid", how="inner")
    df = df[df["objective"] == False]
    return df


def plot_performance(df, performance_metric, extra="", save=True):
    if extra == "":
        df = df.drop_duplicates(subset="uid")
        ylabel = "Proportion of Successfully Evolved Graphs"
        title = "Performance"
        ylim = (0,1.05)
    else:
        ylabel = "Normalized Entropy"
        title = " Diversity"
        ylim = (0,1.2)
    network_sizes = df["network_size"].unique()
    fig, ax = plt.subplots(1, len(network_sizes), figsize=(4*len(network_sizes), 4))
    if len(network_sizes) == 1:
        ax = [ax]
    for i,network_size in enumerate(network_sizes):
        df_ns = df[df["network_size"] == network_size]
        sns.barplot(data=df_ns, x="num_objectives", y=performance_metric, ax=ax[i])
        ax[i].set(title=f"Network Size {network_size}", ylim=ylim)
        ax[i].set(ylabel=None)
    fig.suptitle(f"{extra[1:]}{title} Across Network Size and Number of Constrained Properties")
    fig.supxlabel("Number of Constrained Properties")
    fig.supylabel(ylabel)
    fig.tight_layout()
    if save:
        fig.savefig(f"output/main/{performance_metric}{extra}.png")
    else:
        plt.show()
    plt.close()


def plot_performance_specific(df, network_size, performance_metric, num_obj, hue=None, extra="", save=True):
    if extra == "":
        df = df.drop_duplicates(subset="uid")
        ylabel = "Proportion of Successfully Evolved Graphs"
        title = "Performance"
    else:
        ylabel = "Normalized Entropy"
        title = " Diversity"
    df = df[(df["network_size"] == network_size) & (df["num_objectives"] == num_obj)]
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="objectives", y=performance_metric, hue=hue, ax=ax)
    ax.set(title=f"{extra[1:]}{title} On Network Size {network_size}\nExperiments with {num_obj} Constrained Properties",
           xlabel="Constrained Properties", ylabel=ylabel)
    fig.tight_layout()
    if save:
        fig.savefig(f"output/main/{network_size}/{performance_metric}_{num_obj}{extra}.png")
    else:
        plt.show()
    plt.close()


def diversity_plots(df, property_type, performance_metric, save=True):
    if property_type == "Edge-Weight":
        diversity_properties = topological_properties
    elif property_type == "Topology":
        diversity_properties = edge_weight_properties
    else:
        return
    df1 = df[(df["property"].isin(diversity_properties)) & (df["objective"] == True)]
    keys_to_remove = df1["uid"].unique()
    df = df[~df["uid"].isin(keys_to_remove)]
    df = keep_only_perfect_runs(df)
    df = df[df["property"].isin(diversity_properties)]
    
    extra = f"_{property_type}"
    plot_performance(df, performance_metric, extra, save)
    for network_size in df["network_size"].unique():
        for num_obj in df["num_objectives"].unique():
            plot_performance_specific(df, network_size, performance_metric, 
                                      num_obj, "property", extra, save)


def performance_plots(df, performance_metric, save=True):
    plot_performance(df, performance_metric, save=save)
    df_op = df[[performance_metric, "num_objectives"]].groupby("num_objectives").mean().reset_index()
    not_optimized = df_op[df_op[performance_metric] != 1]["num_objectives"].values
    for network_size in df["network_size"].unique():
        for num_obj in not_optimized:
            plot_performance_specific(df, network_size, performance_metric, num_obj, save=save)


def main():
    df = pd.DataFrame()
    for network_size in get_network_sizes():
        try:
            df_ns = pd.read_pickle(f"output/main/{network_size}/df.pkl")
        except:
            continue
        df_ns["network_size"] = str(network_size)
        if os.path.isfile(f"entropy_{network_size}.json"):
            entropies = json.load(open(f"entropy_{network_size}.json"))
            df_ns["entropy"] = df_ns.apply(lambda row: row["entropy"]/entropies[row["property"]], axis=1)
        df = pd.concat([df, df_ns])

    df["uid"] = df[["network_size", "num_objectives", "exp_num", "rep"]].agg('_'.join, axis=1)
    df["num_objectives"] = df["num_objectives"].astype(int)
    df["exp_num"] = df["exp_num"].astype(int)

    performance_plots(df, "optimized_proportion")
    diversity_plots(df, "Edge-Weight", "entropy")
    diversity_plots(df, "Topology", "entropy")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        if sys.argv[2] == "save":
            save_data(sys.argv[1])
        else:
            print("Please provide a network size and \"save\"")
            print("if the parameter sweep dataframe has not yet been saved for the given network size.")
    else:
        main()