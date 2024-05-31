from itertools import combinations
import json
import os
import sys

from common import (get_network_sizes, get_time_limit, reduce_objective_name,
                   write_nsga_config, write_sbatch, write_scripts_batch)


def main(experiment_type):
    cwd = os.getcwd()
    run_script = []
    analysis_script = []

    if experiment_type == "nsga":
        objectives_to_keep = ["proportion_of_parasitism_pairs", "in_degree_distribution", "average_positive_interactions_strength"]
        crossover_rates = {"low":0.4, "med":0.6, "high":0.8}
        for network_size in get_network_sizes():
            objectives_all = json.load(open(f"objectives_{network_size}.json"))
            objectives = {x:objectives_all[x] for x in objectives_all if x in objectives_to_keep}
            mutation_rates = {"low":1/(network_size**2), "med":5/(network_size**2), "high":10/(network_size**2)}
            popsizes = {"low":10*network_size, "med":50*network_size, "high":100*network_size}
            for num_objectives in range(1, 4):
                combos = list(combinations(objectives, num_objectives))
                for combo in combos:
                    eval_funcs = {x:objectives[x] for x in combo}
                    objective_names = "_".join(reduce_objective_name(x) for x in eval_funcs)
                    for ps in popsizes:
                        for mut in mutation_rates:
                            for co in crossover_rates:
                                full_dir = f"output/{experiment_type}/{objective_names}/{network_size}/p{ps}_m{mut}_c{co}"
                                if not os.path.exists(full_dir):
                                    os.makedirs(full_dir)
                                write_nsga_config(full_dir, mutation_rates[mut], crossover_rates[co], popsizes[ps], 
                                                    2*num_objectives*popsizes[ps], network_size, eval_funcs)
                                write_sbatch(full_dir, objective_names, get_time_limit(network_size), "1gb", 10)
                                run_script.append(f"{cwd}/{full_dir}/job.sb\n")
                                analysis_script.append(f"python3 graph-evolution/replicate_analysis.py {full_dir}\n")
        write_scripts_batch(f"output/{experiment_type}", run_script, analysis_script)

    elif experiment_type == "map-elites":
        return

    else:
        print("Please provide a valid selection scheme.")
        exit()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the selection scheme.")