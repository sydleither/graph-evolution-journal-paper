from itertools import combinations
import json
import os
import sys


EMAIL = "leithers@msu.edu"
USE_ECODE_NODE = True


def reduce_objective_name(objective_name):
    return "".join([x[0] for x in objective_name.split("_")])


def get_time_limit(network_size):
    if network_size == 10:
        return "0-01:00"
    elif network_size == 50:
        return "1-00:00"
    else:
        return "3-00:00"
    

def write_nsga_config(full_dir, mutation_rate, crossover_rate, popsize, num_generations, network_size, eval_funcs):
    full_dir_split = full_dir.split("/")

    config = {
        "data_dir": "/".join(full_dir_split[0:-1]),
        "name": full_dir_split[-1],
        "reps": 1,
        "save_data": 1,
        "plot_data": 0,
        "mutation_rate": mutation_rate,
        "mutation_odds": [1,2,1,2],
        "crossover_odds": [1,2,2],
        "crossover_rate": crossover_rate,
        "weight_range": [-1,1],
        "popsize": popsize,
        "network_size": network_size,
        "num_generations": num_generations,
        "eval_funcs": eval_funcs
    }

    config_path = f"{full_dir}/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def write_sbatch(full_dir, job_name, time_limit, memory_limit, num_replicates):
    job_output_dir = f"{full_dir}/hpcc_out"
    if not os.path.exists(job_output_dir):
        os.mkdir(job_output_dir)
    cwd = os.getcwd()

    with open(f"{full_dir}/job.sb", "w") as f:
        f.write("#!/bin/sh\n")
        if USE_ECODE_NODE:
            f.write("#SBATCH -A ecode\n")
        f.write(f"#SBATCH --mail-type=FAIL\n#SBATCH --mail-user={EMAIL}\n")
        f.write(f"#SBATCH --job-name={job_name}\n#SBATCH -o {job_output_dir}/%A.out\n")
        f.write(f"#SBATCH --time={time_limit}\n")
        f.write(f"#SBATCH --mem-per-cpu={memory_limit}\n")
        f.write(f"#SBATCH --array=0-{num_replicates-1}\n")
        f.write(f"mkdir {full_dir}/${{SLURM_ARRAY_TASK_ID}}\n")
        f.write(f"cd {cwd}\n")
        f.write(f"python3 graph-evolution/main.py {full_dir}/config.json ${{SLURM_ARRAY_TASK_ID}}")


def write_scripts_batch(full_dir, submit_output, analysis_output):
    with open(f"{full_dir}/run_experiments", "w") as f:
        for output_line in submit_output:
            f.write(output_line)

    with open(f"{full_dir}/analyze_experiments", "w") as f:
        for output_line in analysis_output:
            f.write(output_line)


def main(experiment_type):
    run_script = []
    analysis_script = []

    if experiment_type == "parameter_sweep":
        objectives_all = json.load(open("objectives.json"))
        objectives_to_keep = ["clustering_coefficient", "in_degree_distribution", "average_positive_interactions_strength"]
        objectives = {x:objectives_all[x] for x in objectives_all if x in objectives_to_keep}
        crossover_rates = {"low":0.4, "med":0.6, "high":0.8}
        for num_objectives in range(1, 4):
            combos = list(combinations(objectives, num_objectives))
            for combo in combos:
                eval_funcs = {x:objectives[x] for x in combo}
                objective_names = "_".join(reduce_objective_name(x) for x in eval_funcs)
                for network_size in [10, 50, 100]:
                    mutation_rates = {"low":1/(network_size**2), "med":5/(network_size**2), "high":10/(network_size**2)}
                    popsizes = {"low":10*network_size, "med":50*network_size, "high":100*network_size}
                    for ps in popsizes:
                        for mut in mutation_rates:
                            for co in crossover_rates:
                                full_dir = f"output/{experiment_type}/{objective_names}/{network_size}/p{ps}_m{mut}_c{co}"
                                if not os.path.exists(full_dir):
                                    os.makedirs(full_dir)
                                write_nsga_config(full_dir, mutation_rates[mut], crossover_rates[co], popsizes[ps], 
                                                    2*num_objectives*popsizes[ps], network_size, eval_funcs)
                                write_sbatch(full_dir, objective_names, get_time_limit(network_size), "1gb", 10)
                                run_script.append(f"sbatch {full_dir}/job.sb\n")
                                analysis_script.append(f"python3 graph-evolution/replicate_analysis.py {full_dir}\n")
        write_scripts_batch(f"output/{experiment_type}", run_script, analysis_script)

    elif experiment_type == "iterative":
        objectives = json.load(open("objectives.json"))
        for num_objectives in range(1, len(objectives)+1):
            combos = list(combinations(objectives, num_objectives))
            for combo in combos:
                eval_funcs = {x:objectives[x] for x in combo}

    else:
        print("Invalid experiment type.")
        exit()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print("Please provide the experiment type.")