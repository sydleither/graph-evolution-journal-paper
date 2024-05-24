import json
from random import random
import sys

import pandas as pd

sys.path.insert(0, "./graph-evolution")
from organism import Organism


def main(network_size):
    objectives = json.load(open(f"objectives_{network_size}.json"))
    num_samples = 100000
    df_dicts = []
    for _ in range(num_samples):
        org = Organism(network_size, random(), [-1,1])
        row = dict()
        for objective in objectives:
            val = org.getProperty(objective)
            row[objective] = val
        df_dicts.append(row)
    df = pd.DataFrame.from_dict(df_dicts)
    df.to_pickle(f"output/pvalues/df_{network_size}.pkl")


if __name__ == "__main__":
    main(int(sys.argv[1]))