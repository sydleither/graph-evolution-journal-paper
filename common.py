import json
import os


EMAIL = "leithers@msu.edu"
USE_ECODE_NODE = True


def get_network_sizes():
    return [10, 50, 100]


def reduce_objective_name(objective_name):
    return "".join([x[0] for x in objective_name.split("_")])


def get_time_limit(network_size):
    if network_size == 10:
        return "0-01:00"
    elif network_size == 50:
        return "1-00:00"
    else:
        return "3-00:00"
    

def write_config(full_dir, mutation_rate, crossover_rate, popsize, num_generations, network_size, eval_funcs):
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
        f.write(f"#SBATCH --job-name={job_name}\n#SBATCH -o {cwd}/{job_output_dir}/%A.out\n")
        f.write(f"#SBATCH --time={time_limit}\n")
        f.write(f"#SBATCH --mem-per-cpu={memory_limit}\n")
        f.write(f"#SBATCH --array=0-{num_replicates-1}\n")
        f.write(f"cd {cwd}\n")
        f.write(f"mkdir {full_dir}/${{SLURM_ARRAY_TASK_ID}}\n")
        f.write(f"python3 graph-evolution/main.py {full_dir}/config.json ${{SLURM_ARRAY_TASK_ID}}")


def write_scripts_batch(full_dir, submit_output, analysis_output):
    with open(f"{full_dir}/run_experiments", "w") as f:
        for output_line in submit_output:
            f.write(output_line)

    with open(f"{full_dir}/analyze_experiments", "w") as f:
        for output_line in analysis_output:
            f.write(output_line)