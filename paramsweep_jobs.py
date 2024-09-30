'''
This file generates the SLURM jobs to run a parameter sweep.
'''

from itertools import product
import json
import sys

from scipy.stats import qmc

from common import (get_network_sizes, get_time_limit, get_memory_limit, 
                    write_config, write_sbatch, write_scripts_batch, CUR_DIR)


def get_eval_funcs(all_properties):
    idd = all_properties["in_degree_distribution"]
    odd = all_properties["out_degree_distribution"]
    cc = all_properties["clustering_coefficient"]
    apis = all_properties["average_positive_interactions_strength"]
    vpis = all_properties["variance_positive_interactions_strength"]
    pip = all_properties["positive_interactions_proportion"]

    all_eval_funcs = dict()
    all_eval_funcs["cciddodd"] = {"clustering_coefficient":cc, "in_degree_distribution":idd, "out_degree_distribution":odd}
    all_eval_funcs["apispipvpis"] = {"average_positive_interactions_strength":apis,
                                     "positive_interactions_proportion":pip, 
                                     "variance_positive_interactions_strength":vpis}

    return all_eval_funcs


def get_div_funcs():
    all_div_funcs = dict()
    all_div_funcs["cciddodd"] = ["average_positive_interactions_strength", 
                                 "variance_positive_interactions_strength", 
                                 "positive_interactions_proportion"]
    all_div_funcs["apispipvpis"] = ["connectance", "clustering_coefficient"]
    return all_div_funcs


def sample_params(num_samples, param_names, lower_bounds, upper_bounds, ints, seed=42):
    sampler = qmc.LatinHypercube(d=len(lower_bounds), seed=seed)
    unscaled_sample = sampler.random(n=num_samples)
    sample = qmc.scale(unscaled_sample, lower_bounds, upper_bounds).tolist()
    sampled_params = [{param_names[i]:round(s[i]) if ints[i] else round(s[i], 4) for i in range(len(s))} for s in sample]
    return sampled_params


def get_parameter_ranges(network_size):
    params = dict()
    params["crossover_rate"] = {"low":0.4, "high":0.8, "int":False}
    params["mutation_rate"] = {"low":1/(network_size**2), "high":10/(network_size**2), "int":False}
    params["popsize_multiplier"] = {"low":1, "high":5, "int":True}
    params["tournament_probability"] = {"low":0.2, "high":0.8, "int":False}
    params["age_gap"] = {"low":50, "high":500, "int":True}
    return params


def lhs_pramsweep():
    all_div_funcs = get_div_funcs()
    for network_size in get_network_sizes():
        run_script = []
        analysis_script = []
        all_properties = json.load(open(f"objectives_{network_size}.json"))
        all_eval_funcs = get_eval_funcs(all_properties)
        params = get_parameter_ranges(network_size)
        sampled_params = sample_params(100, list(params.keys()),
                                       [params[x]["low"] for x in params],
                                       [params[x]["high"] for x in params], 
                                       [params[x]["int"] for x in params])
        for objective_names in all_eval_funcs:
            eval_funcs = all_eval_funcs[objective_names]
            diversity_funcs = all_div_funcs[objective_names]
            for i,exp_params in enumerate(sampled_params):
                exp_dir = f"output/paramsweep/{network_size}/{objective_names}/params{i}"
                num_gen = 100*network_size
                popsize = 10*exp_params["popsize_multiplier"]*network_size
                write_config(full_dir=exp_dir, track_diversity_over=diversity_funcs, tracking_frequency=10,
                             network_size=network_size, num_generations=num_gen, eval_funcs=eval_funcs,
                             age_gap=exp_params["age_gap"], mutation_rate=exp_params["mutation_rate"],
                             crossover_rate=exp_params["crossover_rate"], popsize=popsize,
                             tournament_probability=exp_params["tournament_probability"])
                write_sbatch(exp_dir, objective_names, get_time_limit(network_size), get_memory_limit(network_size), 3)
                run_script.append(f"sbatch {CUR_DIR}/{exp_dir}/job.sb\n")
                analysis_script.append(f"python3 graph-evolution/replicate_analysis.py {exp_dir}\n")
        write_scripts_batch(f"output/paramsweep/{network_size}", run_script, analysis_script)


def get_parameter_values(network_size):
    params = dict()
    params["crossover_rate"] = {"low":0.4, "high":0.8, "med":0.6}
    params["mutation_rate"] = {"low":1/(network_size**2), "high":10/(network_size**2), "med":5/(network_size**2)}
    params["popsize"] = {"low":10*network_size, "high":50*network_size, "med":30*network_size}
    params["tournament_probability"] = {"low":0.25, "high":0.75, "med":0.5}
    params["age_gap"] = {"low":10*network_size, "high":30*network_size, "med":20*network_size}
    return params


def trad_pramsweep():
    all_div_funcs = get_div_funcs()
    for network_size in get_network_sizes():
        all_properties = json.load(open(f"objectives_{network_size}.json"))
        all_eval_funcs = get_eval_funcs(all_properties)
        params = get_parameter_values(network_size)
        param_combos = list(product(["low", "med", "high"], repeat=len(params)))
        for objective_names in all_eval_funcs:
            run_script = []
            analysis_script = []
            eval_funcs = all_eval_funcs[objective_names]
            diversity_funcs = all_div_funcs[objective_names]
            for param_combo in param_combos:
                cr = param_combo[0]
                mr = param_combo[1]
                p = param_combo[2]
                tp = param_combo[3]
                ag = param_combo[4]
                exp_dir = f"output/paramsweep/{network_size}/{objective_names}/{cr}_{mr}_{p}_{tp}_{ag}"
                num_gen = 100*network_size
                write_config(full_dir=exp_dir, track_diversity_over=diversity_funcs, tracking_frequency=10,
                             network_size=network_size, num_generations=num_gen, eval_funcs=eval_funcs,
                             age_gap=params["age_gap"][ag], mutation_rate=params["mutation_rate"][mr], 
                             crossover_rate=params["crossover_rate"][cr], popsize=params["popsize"][p],
                             tournament_probability=params["tournament_probability"][tp])
                write_sbatch(exp_dir, objective_names, get_time_limit(network_size), get_memory_limit(network_size), 3)
                run_script.append(f"sbatch {CUR_DIR}/{exp_dir}/job.sb\n")
                analysis_script.append(f"python3 graph-evolution/replicate_analysis.py {exp_dir}\n")
            write_scripts_batch(f"output/paramsweep/{network_size}/{objective_names}", run_script, analysis_script)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "trad":
            trad_pramsweep()
        elif sys.argv[1] == "lhs":
            lhs_pramsweep()
    else:
        print("Please specific parameter sweep type: trad or lhs.")