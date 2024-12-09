import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from paramsweep_jobs import get_div_funcs, get_parameter_values


colors = ["#509154", "#A9561E", "#77BCFD", "#B791D4", "#EEDD5D", 
          "#738696", "#24BCA8", "#D34A4F", "#8D81FE", "#FDA949"]
sns.set_palette(sns.color_palette(colors))


def save_data(network_size):
    all_div_funcs = get_div_funcs()
    exp_dir = f"output/paramsweep/{network_size}"
    df = pd.DataFrame()
    for objectives in os.listdir(exp_dir):
        objective_path = f"{exp_dir}/{objectives}"
        if os.path.isfile(objective_path):
            continue
        print(f"Saving {objectives}...")
        for parami in os.listdir(objective_path):
            param_path = f"{objective_path}/{parami}"
            if os.path.isfile(param_path):
                continue
            for replicate in os.listdir(param_path):
                rep_path = f"{param_path}/{replicate}"
                if os.path.isfile(rep_path) or replicate == "hpcc_out":
                    continue
                if len(os.listdir(rep_path)) == 0:
                    print(f"Data not found: {objectives} {parami} {replicate}")
                    continue
                fitnesses = pd.read_pickle(f"{rep_path}/fitness_log.pkl")
                fitnesses = {k:v[-1] for k,v in fitnesses.items()}
                objective_properties = list(fitnesses.keys())
                properties_of_interest = all_div_funcs[objectives] + objective_properties
                df_i = pd.read_csv(f"{rep_path}/diversity.csv")
                df_i = df_i.loc[df_i["property"].isin(properties_of_interest)]
                df_i["param_set"] = parami
                df_i["objectives"] = objectives
                df_i["rep"] = replicate
                df_i["objective"] = False
                df_i.loc[df_i["property"].isin(objective_properties), "objective"] = True
                df = pd.concat([df, df_i])
    df = df.reset_index()
    pd.to_pickle(df, f"{exp_dir}/df.pkl")


def keep_only_perfect_runs(df):
    key = ["objectives", "param_set"]
    avg_performance = df[key+["optimized_proportion"]].groupby(key).mean().reset_index()
    perfect_runs = avg_performance.loc[avg_performance["optimized_proportion"] == 1][key]
    df = df.merge(perfect_runs, on=key, how="inner")
    df = df[df["objective"] == False]
    return df


def plot_parameter_performance(df, network_size, param_names, performance_metric, save=True):
    df = df.drop_duplicates(subset=["objectives", "rep", "param_set"])
    num_params = len(param_names)
    fig, ax = plt.subplots(1, num_params, figsize=(8*num_params,8))
    for p in range(num_params):
        sns.barplot(data=df, x="objectives", y=performance_metric, hue=param_names[p], ax=ax[p])
    fig.tight_layout()
    if save:
        plt.savefig(f"output/paramsweep/{network_size}/parameter-{performance_metric}.png")
    else:
        plt.show()
    plt.close()


def plot_parameter_diversity(df, network_size, param_names, performance_metric, save=True):
    df = keep_only_perfect_runs(df)
    objectives = df["objectives"].unique()
    num_objectives = len(objectives)
    num_params = len(param_names)
    fig, ax = plt.subplots(num_objectives, num_params, figsize=(8*num_params,8*num_objectives))
    for o in range(num_objectives):
        df_o = df.loc[df["objectives"] == objectives[o]]
        for p in range(num_params):
            sns.barplot(data=df_o, x="property", y=performance_metric, hue=param_names[p], ax=ax[o][p])
            ax[o][p].set_title(objectives[o])
    fig.tight_layout()
    if save:
        plt.savefig(f"output/paramsweep/{network_size}/parameter-{performance_metric}.png")
    else:
        plt.show()
    plt.close()


def plot_two_params(df, network_size, param1, param2, performance_metric, save=True):
    if performance_metric == "optimized_proportion":
        df = df.drop_duplicates(subset=["objectives", "rep", "param_set", "property"])
        df = df.loc[df["objective"] == True]
    else:
        df = keep_only_perfect_runs(df)

    objectives = df["objectives"].unique()
    properties = df["property"].unique()
    num_objectives = len(objectives)
    num_properties = len(properties)
    fig, ax = plt.subplots(num_objectives, num_properties, figsize=(8*num_properties,8*num_objectives))
    for o in range(num_objectives):
        for p in range(num_properties):
            df_op = df.loc[(df["objectives"] == objectives[o]) & (df["property"] == properties[p])]
            if len(df_op) == 0:
                continue
            df_op_avg = df_op.groupby([param1, param2])[performance_metric].mean().unstack()
            sns.heatmap(df_op_avg, fmt="g", annot=True, cmap="Greens_r", ax=ax[o][p])
            ax[o][p].set_title(f"{objectives[o]} {properties[p]}")
    fig.patch.set_facecolor("#ffd1df")
    fig.suptitle(performance_metric)
    fig.tight_layout()
    if save:
        plt.savefig(f"output/paramsweep/{network_size}/heatmap-{param1}-{param2}-{performance_metric}.png")
    else:
        plt.show()
    plt.close()


def plot_best_params(df, network_size, param_names, performance_metric, save=True):
    df = keep_only_perfect_runs(df)
    convert_for_order = {"low":"2low", "med":"1med", "high":"0high"}
    for p,param in enumerate(param_names):
        df[param] = df["param_set"].str.split("_").str[p].map(convert_for_order)

    objectives = df["objectives"].unique()
    num_objectives = len(objectives)
    for o in range(num_objectives):
        df_o = df.loc[df["objectives"] == objectives[o]]
        properties = df_o["property"].unique()
        num_properties = len(properties)
        fig, ax = plt.subplots(1, num_properties, figsize=(8*num_properties,8))
        for p in range(num_properties):
            df_op = df_o.loc[df_o["property"] == properties[p]]
            if len(df_op) == 0:
                continue
            df_op_long1 = pd.melt(df_op, id_vars="param_set", value_vars=param_names, var_name="param").drop_duplicates()
            df_op_long2 = pd.melt(df_op, id_vars="param_set", value_vars=[performance_metric], value_name="measure")
            df_op_long = df_op_long1.merge(df_op_long2[["param_set", "measure"]], on="param_set")
            df_op_long = df_op_long.groupby(["param_set", "param", "value"]).mean().reset_index()
            df_op_long_rank = df_op_long[["param_set", "measure"]].drop_duplicates()
            df_op_long_rank["rank"] = df_op_long_rank["measure"].rank(method="min", ascending=False)
            df_op_long = df_op_long_rank.merge(df_op_long, on=["param_set", "measure"])
            df_op_long = df_op_long.loc[df_op_long["rank"] <= 10]
            df_op_long["rank"] = df_op_long["rank"].astype(str)
            rank_order = [str(float(x)) for x in range(1,11)]
            df_op_long = df_op_long.sort_values(by=["value"])
            sns.pointplot(data=df_op_long, x="param", y="value", hue="measure", hue_order=rank_order, 
                          order=param_names, errorbar=None, dodge=True, palette="Greens_r", ax=ax[p])
            ax[p].set_title(properties[p])
            ax[p].set_facecolor("#ffd1df")
        fig.suptitle(f"{objectives[o]} {performance_metric}")
        fig.tight_layout()
        if save:
            plt.savefig(f"output/paramsweep/{network_size}/pointplot-{objectives[o]}-{performance_metric}.png")
        else:
            plt.show()
        plt.close()


def score_params(df, param_names, performance_metric):
    if performance_metric == "optimized_proportion":
        df = df.drop_duplicates(subset=["objectives", "rep", "param_set", "property"])
        df = df.loc[df["objective"] == True]
        key = ["objectives"]
    else:
        df = keep_only_perfect_runs(df)
        key = ["objectives", "property"]
    df = df[key+["param_set", performance_metric]+param_names]

    df = df.groupby(key+param_names+["param_set"]).mean().reset_index()
    df["index"] = df.index
    df_grp = df[key+[performance_metric]].groupby(key).rank(method="min", ascending=False).reset_index()
    df_grp = df_grp.rename(columns={performance_metric:"rank"})
    df = df.merge(df_grp, on=["index"])
    df = df.loc[df["rank"] <= 10]
    df_final = df.groupby(["param_set"]+param_names).count().reset_index()
    df_final = df_final[["param_set", "rank"]+param_names].rename(columns={"rank":"count"})
    print(df_final.sort_values(by=["count"], ascending=False).loc[df_final["count"] > 1])


def main(network_size):
    try:
        df = pd.read_pickle(f"output/paramsweep/{network_size}/df.pkl")
    except:
        print("Please save the dataframe.")
        exit()

    params = get_parameter_values(int(network_size))
    param_names = list(params.keys())
    for p,param in enumerate(param_names):
        df[param] = df["param_set"].str.split("_").str[p].map(params[param])

    plot_best_params(df, network_size, param_names, "entropy")
    for diversity_measurement in ["spread", "entropy", "uniformity", "unique_types", "optimized_proportion"]:
        print(diversity_measurement)
        if diversity_measurement == "optimized_proportion":
            plot_parameter_performance(df, network_size, param_names, diversity_measurement)
        else:
            plot_parameter_diversity(df, network_size, param_names, diversity_measurement)
            plot_best_params(df, network_size, param_names, diversity_measurement)
        score_params(df, param_names, diversity_measurement)


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