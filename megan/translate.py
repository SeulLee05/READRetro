import os
import sys
root = os.path.abspath('.')
if root not in sys.path:
    sys.path.insert(0, root)

import torch
from rdkit import Chem

from megan.split.basic_splits import DefaultSplit
from megan.feat.megan_graph import MeganTrainingSamplesFeaturizer
from megan.model.megan import Megan
from megan.model.megan_utils import get_base_action_masks, RdkitCache
from megan.feat.utils import fix_explicit_hs
from megan.model.beam_search import beam_search


def filter_invalid(generations, scores):
    filtered_generations, filtered_scores = [], []
    for generation, score in zip(generations, scores):
        mol = Chem.MolFromSmiles(generation)
        if mol is None:
            continue
        filtered_generations.append(Chem.MolToSmiles(mol))
        filtered_scores.append(score)
    
    return filtered_generations, filtered_scores


def remap_to_canonical(input_smiles):
    input_mol = Chem.MolFromSmiles(input_smiles)

    map2map = {}
    for i, a in enumerate(input_mol.GetAtoms()):
        map2map[int(a.GetAtomMapNum())] = i + 1
        a.SetAtomMapNum(i + 1)

    return input_mol


def prepare_megan(cuda=True,
                  path='megan/saved_models/biochem.pt',
                  vocab_path='megan/saved_models/clean'):
    featurizer = MeganTrainingSamplesFeaturizer(n_jobs=-1, max_n_steps=32,
                                                split=DefaultSplit(),
                                                action_order='bfs_randat')
    action_vocab = featurizer.get_actions_vocabulary(vocab_path)
    
    device = torch.device("cuda") if cuda else torch.device("cpu")
    checkpoint = torch.load(path)
    model = Megan(n_atom_actions=action_vocab['n_atom_actions'], n_bond_actions=action_vocab['n_bond_actions'],
                  prop2oh=action_vocab['prop2oh']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    base_action_masks = get_base_action_masks(200 + 1, action_vocab=action_vocab)

    return model, action_vocab, base_action_masks


def run_megan(model, action_vocab, base_action_masks, smi):
    mol = remap_to_canonical(smi)
    mol = fix_explicit_hs(mol)
    rdkit_cache = RdkitCache(props=action_vocab['props'])

    with torch.no_grad():
        result = beam_search([model], [mol], rdkit_cache=rdkit_cache, max_steps=16,
                                            beam_size=10, batch_size=10,
                                            base_action_masks=base_action_masks, max_atoms=200,
                                            reaction_types=None,
                                            action_vocab=action_vocab)[0]
    reactants = [r['final_smi_unmapped'] for r in result]
    scores = [r['prob'] for r in result]
    reactants, scores = filter_invalid(reactants, scores)
    result = {'reactants': reactants, 'scores': scores}

    result['templates'] = [None for _ in range(len(result['scores']))]
    result['retrieved'] = [False for _ in range(len(result['scores']))]
    return result


if __name__ == "__main__":
    model, action_vocab, base_action_masks = prepare_megan()
    # smi = 'N[C@@H](CNC(=O)C(=O)O)C(=O)O'
    smi = 'C=CC(C)(C)[C@@]12C[C@H]3c4nc5ccccc5c(=O)n4[C@H](C)C(=O)N3C1N(C(C)=O)c1ccccc12'
    print(run_megan(model, action_vocab, base_action_masks, smi))
