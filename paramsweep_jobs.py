'''
This file generates the SLURM jobs to run a parameter sweep.
'''

import json

from scipy.stats import qmc

from common import (get_network_sizes, get_time_limit, write_config,
                    write_sbatch, write_scripts_batch, CUR_DIR)


def sample_params(num_samples, param_names, lower_bounds, upper_bounds, ints):
    sampler = qmc.LatinHypercube(d=len(lower_bounds))
    unscaled_sample = sampler.random(n=num_samples)
    sample = qmc.scale(unscaled_sample, lower_bounds, upper_bounds).tolist()
    sampled_params = [{param_names[i]:int(s[i]) if ints[i] else round(s[i], 4) for i in range(len(s))} for s in sample]
    return sampled_params


def get_parameters(network_size):
    params = dict()
    params["crossover_rate"] = {"low":0.4, "high":0.8, "int":False}
    params["mutation_rate"] = {"low":1/(network_size**2), "high":10/(network_size**2), "int":False}
    params["popsize"] = {"low":10*network_size, "high":100*network_size, "int":True}
    params["tournament_probability"] = {"low":0.2, "high":0.8, "int":False}
    params["age_gap"] = {"low":50, "high":500, "int":True}
    return params


def get_eval_funcs(all_properties):
    idd = all_properties["in_degree_distribution"]
    cc = all_properties["clustering_coefficient"]
    apis = all_properties["average_positive_interactions_strength"]
    pip = all_properties["positive_interactions_proportion"]

    all_eval_funcs = dict()
    all_eval_funcs["ccidd"] = {"clustering_coefficient":cc, "in_degree_distribution":idd}
    all_eval_funcs["ccapis"] = {"clustering_coefficient":cc, "average_positive_interactions_strength":apis}
    all_eval_funcs["iddpip"] = {"in_degree_distribution":idd, "positive_interactions_proportion":pip}
    all_eval_funcs["apispip"] = {"average_positive_interactions_strength":apis, "positive_interactions_proportion":pip}

    return all_eval_funcs


def main():
    run_script = []
    analysis_script = []
    for network_size in get_network_sizes():
        all_properties = json.load(open(f"objectives_{network_size}.json"))
        all_eval_funcs = get_eval_funcs(all_properties)
        params = get_parameters(network_size)
        sampled_params = sample_params(100, list(params.keys()), [params[x]["low"] for x in params], 
                                       [params[x]["high"] for x in params], [params[x]["int"] for x in params])
        for objective_names in all_eval_funcs:
            eval_funcs = all_eval_funcs[objective_names]
            diversity_funcs = [x for x in all_properties if x not in eval_funcs and not x.endswith("distribution")]
            for i,exp_params in enumerate(sampled_params):
                exp_dir = f"output/paramsweep/{network_size}/{objective_names}/params{i}"
                write_config(full_dir=exp_dir, track_diversity_over=diversity_funcs,
                                network_size=network_size, num_generations=5000, eval_funcs=eval_funcs,
                                age_gap=exp_params["age_gap"], mutation_rate=exp_params["mutation_rate"],
                                crossover_rate=exp_params["crossover_rate"], popsize=exp_params["popsize"],
                                tournament_probability=exp_params["tournament_probability"])
                write_sbatch(exp_dir, objective_names, get_time_limit(network_size), "2gb", 3)
                run_script.append(f"{CUR_DIR}/{exp_dir}/job.sb\n")
                analysis_script.append(f"python3 graph-evolution/replicate_analysis.py {exp_dir}\n")
    write_scripts_batch("output/paramsweep", run_script, analysis_script)


if __name__ == "__main__":
    main()