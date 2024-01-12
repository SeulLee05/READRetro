import os
import numpy as np
import logging
from retro_star.alg.mol_tree import MolTree


def molstar(target_mol, starting_mols, expand_fn, value_fn, iterations):
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn
    )

    if not mol_tree.succ:
        for i in range(iterations):
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                # logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            result = expand_fn(m_next.mol)
            if result is not None and len(result['scores']) > 0:
                reactants = result['reactants']
                scores = result['scores']
                retrieved = result['retrieved']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                # import pdb; pdb.set_trace()

                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    import pdb; pdb.set_trace()
                    templates = result['templates'] # need to fix for integrity

                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                
                # from BioNavi-NP
                mol_tree.expand(m_next, reactant_lists, costs, templates, retrieved)

            else:
                mol_tree.expand(m_next, None, None, None, None)

    routes = []
    if mol_tree.succ:
        # import pdb; pdb.set_trace()
        best_route = mol_tree.get_best_route()
        routes = mol_tree.get_routes()
        routes = sorted(routes, key=lambda x: x.total_cost)
        assert best_route is not None

    return mol_tree.succ, routes
