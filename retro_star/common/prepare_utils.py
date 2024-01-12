import pickle
import pandas as pd
from retro_star.alg import molstar


def prepare_starting_molecules(filename):
    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    # logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols


def prepare_molstar_planner(expansion_handler, value_fn, starting_mols, iterations):
    plan_handle = lambda x: molstar(
        target_mol=x,
        starting_mols=starting_mols,
        expand_fn=expansion_handler,
        value_fn=value_fn,
        iterations=iterations,
    )
    
    return plan_handle
