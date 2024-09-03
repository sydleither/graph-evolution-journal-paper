import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import reduce_objective_name
from paramsweep_jobs import get_parameters


def save_data(network_size):
    exp_dir = f"output/paramsweep/{network_size}"
    df = pd.DataFrame()
    for objectives in os.listdir(exp_dir):
        print(objectives)
        objective_path = f"{exp_dir}/{objectives}"
        if os.path.isfile(objective_path):
            continue
        for parami in os.listdir(objective_path):
            param_path = f"{objective_path}/{parami}"
            for replicate in os.listdir():
                rep_path = f"{param_path}/{replicate}"
                if os.path.isfile(rep_path) or replicate == "hpcc_out" or len(os.listdir(rep_path)) == 0:
                    continue
                fitnesses = pd.read_pickle(f"{rep_path}/fitness_log.pkl")
                fitnesses = {k:v[-1] for k,v in fitnesses.items()}
                df_i = pd.read_csv(f"{rep_path}/diversity.csv")
                df_i["param_set"] = parami
                df_i["objectives"] = objectives
                df_i["rep"] = replicate
                for objective in fitnesses:
                    df_i.loc[df_i["property"] == objective, "optimized"] = "yes" if fitnesses[objective] == 0 else "no"
                df = pd.concat([df, df_i])
    df = df.reset_index()
    pd.to_pickle(df, f"{exp_dir}/df.pkl")


def main(network_size):
    try:
        df = pd.read_pickle(f"output/paramsweep/{network_size}/df.pkl")
    except:
        print("Please save the dataframe.")
        exit()
    parameters_swept = list(get_parameters(network_size).keys())
    diversity_measures = ["spread", "entropy"]


if __name__ == "__main__":
    if len(sys.argv) == 3:
        if sys.argv[3] == "save":
            save_data(sys.argv[1])
        else:
            print("Please provide a network size and \"save\"")
            print("if the parameter sweep dataframe has not yet been saved for the given network size.")
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide only a network size argument, if the dataframe has been saved.")