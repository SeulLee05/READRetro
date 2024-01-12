import numpy as np
from queue import Queue
from copy import deepcopy
from retro_star.alg.mol_node import MolNode
from retro_star.alg.reaction_node import ReactionNode
from retro_star.alg.syn_route import SynRoute


class MolTree:
    def __init__(self, target_mol, known_mols, value_fn, zero_known_value=True):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.value_fn = value_fn
        self.zero_known_value = zero_known_value
        self.mol_nodes = []
        self.reaction_nodes = []

        self.root = self._add_mol_node(target_mol, None)
        self.succ = target_mol in known_mols
        self.search_status = 0

        # if self.succ:
        #     logging.info('Synthesis route found: target in starting molecules')

    def _add_mol_node(self, mol, parent, retrieved=False):
        is_known = mol in self.known_mols

        init_value = self.value_fn(mol, retrieved)

        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known,
            zero_known_value=self.zero_known_value
        )
        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)

        return mol_node

    def _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors, retrieved):
        assert cost >= 0

        for mol in mols:
            if mol in ancestors:
                return

        reaction_node = ReactionNode(parent, cost, template)
        for mol in mols:
            self._add_mol_node(mol, reaction_node, retrieved)
        reaction_node.init_values()
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def expand(self, mol_node, reactant_lists, costs, templates, retrieved):
        assert not mol_node.is_known and not mol_node.children

        if costs is None:      # No expansion results
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ

        assert mol_node.open
        ancestors = mol_node.get_ancestors()
        for i in range(len(costs)):
            self._add_reaction_and_mol_nodes(costs[i], reactant_lists[i],
                                             mol_node, templates[i], ancestors, retrieved[i])

        if len(mol_node.children) == 0:      # No valid expansion results
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ

        v_delta = mol_node.init_values()
        if mol_node.parent:
            mol_node.parent.backup(v_delta, from_mol=mol_node.mol)

        if not self.succ and self.root.succ:
            self.succ = True

        return self.succ

    def get_best_route(self):
        if not self.succ:
            return None

        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )

        mol_queue = Queue()
        mol_queue.put(self.root)
        while not mol_queue.empty():
            mol = mol_queue.get()
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue

            best_reaction = None
            for reaction in mol.children:
                if reaction.succ:
                    if best_reaction is None or \
                            reaction.succ_value < best_reaction.succ_value:
                        best_reaction = reaction
            assert best_reaction.succ_value == mol.succ_value

            reactants = []
            for reactant in best_reaction.children:
                mol_queue.put(reactant)
                reactants.append(reactant.mol)

            syn_route.add_reaction(
                mol=mol.mol,
                value=mol.succ_value,
                template=best_reaction.template,
                reactants=reactants,
                cost=best_reaction.cost
            )

        return syn_route

    # from BioNavi-NP
    def get_routes(self):
        if not self.succ:
            return None

        routes = []
        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )
        routes.append(syn_route)

        mol_queue = Queue()
        mol_queue.put((syn_route, self.root))
        while not mol_queue.empty():
            syn_route, mol = mol_queue.get()
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue

            best_reaction = None
            all_reactions = []
            for reaction in mol.children:
                if reaction.succ:
                    all_reactions.append(reaction)
                    if best_reaction is None or \
                            reaction.succ_value < best_reaction.succ_value:
                        best_reaction = reaction
            # assert best_reaction.succ_value == mol.succ_value

            syn_route_template = None
            if len(all_reactions) > 1:
                syn_route_template = deepcopy(syn_route)

            # best reaction
            reactants = []
            for reactant in best_reaction.children:
                mol_queue.put((syn_route, reactant))
                reactants.append(reactant.mol)

            syn_route.add_reaction(
                mol=mol.mol,
                value=mol.succ_value,
                template=best_reaction.template,
                reactants=reactants,
                cost=best_reaction.cost
            )

            # other reactions
            if len(all_reactions) > 1:
                for reaction in all_reactions:
                    if reaction == best_reaction:
                        continue

                    syn_route = deepcopy(syn_route_template)
                    routes.append(syn_route)

                    reactants = []
                    for reactant in reaction.children:
                        mol_queue.put((syn_route, reactant))
                        reactants.append(reactant.mol)

                    syn_route.add_reaction(
                        mol=mol.mol,
                        value=mol.succ_value,   # might be incorrect
                        template=reaction.template,
                        reactants=reactants,
                        cost=reaction.cost
                    )

            # print('succ value: %.2f | total cost: %.2f | length: %d' %
            #       (syn_route.succ_value, syn_route.total_cost, syn_route.length))

        return routes
