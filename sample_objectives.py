'''
This file generates a specified sample of random graphs.
A full sample will save each random graph's properties.
    Example: python3 sample_objectives.py full 10 500
A limited sample will only save whether the random graph's property matched the objective value or not.
    Example: python3 sample_objectives.py limited 10 500
A slurm job config can also be generated to run the sampling.
    Example: python3 sample_objectives.py job

Full sample can be used to visualize distributions of properties of random graphs.
Limited sample can be used to glean how difficult the objective value will be to evolve.
This is done using sample_objectives_analysis.py.
'''

import csv
import json
import os
from random import random
import sys

from common import write_sbatch

sys.path.insert(0, "./graph-evolution")
from organism import Organism


output_dir = "output/sample_objectives"


def limited_sample(org, objectives):
    return [1 if org.getProperty(objective) == objectives[objective] else 0 for objective in objectives]


def full_sample(org, objectives):
    return [org.getProperty(objective) for objective in objectives]


def main(network_size, num_samples, limited):
    objectives = json.load(open(f"objectives_{network_size}.json"))
    if "connectance" not in objectives:
        objectives["connectance"] = 0.6
    
    if limited:
        sample_func = limited_sample
        output_name = f"{network_size}_limited.csv"
    else:
        sample_func = full_sample
        output_name = f"{network_size}_full.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/{output_name}", "w") as f:
        f_writer = csv.writer(f)
        f_writer.writerow(objectives)
        for _ in range(num_samples):
            org = Organism(network_size, random(), [-1,1])
            row = sample_func(org, objectives)
            f_writer.writerow(row)


def generate_job():
    run_line = "python3 sample_objectives.py ${1} ${2} ${3}"
    write_sbatch(output_dir, "sample", "0-00:30", "1gb", 1, run_line)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        try:
            network_size = int(sys.argv[2])
            num_samples = int(sys.argv[3])
        except:
            print("Error converting network size and number of sample arguments into integers.")
            exit()
        if sys.argv[1] == "full":
            main(network_size, num_samples, limited=False)
        elif sys.argv[1] == "limited":
            main(network_size, num_samples, limited=True)
        else:
            print("Please provide the sample type: full or limited.")
    elif len(sys.argv) == 2:
        if sys.argv[1] == "job":
            generate_job()
        else:
            print("Please specify the job argument.")
    else:
        print("Please read of the top of this file for usage instructions.")