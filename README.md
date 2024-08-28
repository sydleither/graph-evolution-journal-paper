# graph-evolution-journal-paper

Experiments for the journal paper about the graph-evolution tool.

Pipeline

- Create objectives and target values with create_objectives.py

- Check that the objectives are valid and reachable with check_objectives.py

- Quantify the difficultly of reaching those objectives with sample_objectives.py

- Find the best model parameters for a subset of the objectives with paramsweep_jobs.py

    - This will generate scripts for running the parameter sweep using graph-evolution/main.py on the MSU HPCC

    - Analyze the results with paramsweep_analysis.py

In progress

- Actual main experiments for paper (iteratively add objectives to runs)