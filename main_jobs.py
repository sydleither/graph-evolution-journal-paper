'''
This file generates the SLURM jobs to run the main experiments.
'''

from itertools import combinations
import json

from common import (get_network_sizes, get_time_limit, get_memory_limit, 
                    write_config, write_sbatch, write_scripts_batch, HPCC_DIR)


def get_diversity_funcs(network_size, all_properties, objectives):
    div_funcs = [x for x in all_properties if x not in objectives]
    div_funcs = ["connectance" if x.endswith("degree_distribution") else x for x in div_funcs]
    if "connectance" in div_funcs and network_size == 10:
        div_funcs.remove("connectance")
    return div_funcs


def main():
    for network_size in get_network_sizes():
        all_properties = json.load(open(f"objectives_{network_size}.json"))
        run_script = []
        analysis_script = []
        for num_obj in range(1, len(all_properties)+1):
            combos = list(combinations(all_properties, num_obj))
            for c,combo in enumerate(combos):
                if "out_degree_distribution" in combo and "in_degree_distribution" not in combo:
                    continue
                diversity_funcs = get_diversity_funcs(network_size, all_properties, combo)
                exp_dir = f"output/main/{network_size}/{num_obj}/{c}"
                num_gen = 100*network_size
                eval_funcs = {combo[x]:all_properties[combo[x]] for x in range(len(combo))}
                write_config(full_dir=exp_dir, track_diversity_over=diversity_funcs, tracking_frequency=10,
                             network_size=network_size, num_generations=num_gen, eval_funcs=eval_funcs,
                             crossover_rate=0.4, popsize=50*network_size, age_gap=10*network_size,
                             tournament_probability=0.5, mutation_rate=1/(network_size**2))
                write_sbatch(exp_dir, num_obj+"_"+c, get_time_limit(network_size), get_memory_limit(network_size), 10)
                run_script.append(f"sbatch {HPCC_DIR}/{exp_dir}/job.sb\n")
                analysis_script.append(f"python3 graph-evolution/replicate_analysis.py {exp_dir}\n")
        write_scripts_batch(f"output/main/{network_size}", run_script, analysis_script)


if __name__ == "__main__":
    main()