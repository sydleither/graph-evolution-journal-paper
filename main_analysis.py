import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from main_jobs import get_diversity_funcs


colors = ["#509154", "#A9561E", "#77BCFD", "#B791D4", "#EEDD5D", 
          "#738696", "#24BCA8", "#D34A4F", "#8D81FE", "#FDA949"]
sns.set_palette(sns.color_palette(colors))


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
                    print(f"sbatch output/main/{network_size}/{num_objectives}/{num_exp}/job.sb")
                    continue
                fitnesses = pd.read_pickle(f"{rep_path}/fitness_log.pkl")
                fitnesses = {k:v[-1] for k,v in fitnesses.items()}
                objective_properties = list(fitnesses.keys())
                div_funcs = get_diversity_funcs(network_size, all_properties, objective_properties)
                properties_of_interest = div_funcs + objective_properties
                df_i = pd.read_csv(f"{rep_path}/diversity.csv")
                df_i = df_i.loc[df_i["property"].isin(properties_of_interest)]
                df_i["exp_num"] = num_exp
                df_i["num_objectives"] = num_objectives
                df_i["rep"] = replicate
                df_i["objective"] = False
                df_i.loc[df_i["property"].isin(objective_properties), "objective"] = True
                df = pd.concat([df, df_i])
    df = df.reset_index()
    pd.to_pickle(df, f"{exp_dir}/df.pkl")


def keep_only_perfect_runs(df):
    key = ["num_objectives", "param_set"]
    avg_performance = df[key+["optimized_proportion"]].groupby(key).mean().reset_index()
    perfect_runs = avg_performance.loc[avg_performance["optimized_proportion"] == 1][key]
    df = df.merge(perfect_runs, on=key, how="inner")
    df = df[df["objective"] == False]
    return df


def main(network_size):
    try:
        df = pd.read_pickle(f"output/main/{network_size}/df.pkl")
    except:
        print("Please save the dataframe.")
        exit()

    if os.path.isfile(f"entropy_{network_size}.json"):
        entropies = json.load(open(f"entropy_{network_size}.json"))
        df["entropy"] = df.apply(lambda row: row["entropy"]/entropies[row["property"]], axis=1)

    print(df)


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