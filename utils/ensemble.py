import os
import sys
root = os.path.abspath('.')
if root not in sys.path:
    sys.path.insert(0, root)

from time import time
import numpy as np
from retroformer.translate import prepare_retroformer, run_retroformer
from g2s.translate import prepare_g2s, run_g2s


def prepare_ensemble(cuda=True, beam_size=10, expansion_topk=10,
                     retroformer_path='retroformer/saved_models/biochem.pt',
                     retroformer_vocab_path='retroformer/saved_models/vocab_share.pk',
                     g2s_path='g2s/saved_models/biochem.pt',
                     g2s_vocab_path='g2s/saved_models/vocab_smiles.txt'):
    model_retroformer, args_retroformer = prepare_retroformer(cuda, beam_size, expansion_topk, retroformer_path, retroformer_vocab_path)
    model_g2s, args_g2s, vocab, vocab_tokens, device = prepare_g2s(cuda, 200, expansion_topk, g2s_path, g2s_vocab_path)
    return model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, device


def run_ensemble(x, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                 vocab, vocab_tokens, device, expansion_topk=10):
    if model_type == 'ensemble':
        result = run_retroformer(model_retroformer, args_retroformer, x)
        _result = run_g2s(model_g2s, args_g2s, x, vocab, vocab_tokens, device)
        result['reactants'].extend(_result['reactants'])
        result['scores'].extend(_result['scores'])
        
    elif model_type == 'retroformer':
        result = run_retroformer(model_retroformer, args_retroformer, x)
    elif model_type == 'g2s':
        result = run_g2s(model_g2s, args_g2s, x, vocab, vocab_tokens, device)
    else:
        print("Wrong model type")   # Not the case
        
    reactants, scores = [], []
    for reactant, score in zip(result['reactants'], result['scores']):
        if reactant not in reactants:
            reactants.append(reactant)
            scores.append(score)
    result['reactants'], result['scores'] = reactants, scores
    
    ordering = np.argsort(result['scores'])[::-1][:expansion_topk]
    result['reactants'] = list(np.array(result['reactants'])[ordering])
    result['scores'] = list(np.array(result['scores'])[ordering])
    
    if len(result['scores']) > 0:
        result['scores'] = list(np.exp(np.array(result['scores'])) / np.sum(np.exp(np.array(result['scores']))))
    result['retrieved'] = [False for _ in result['scores']]
    result['templates'] = [None for _ in result['scores']]
    return result
        

def run_ensemble_singlestep(smi_list, model_retroformer, model_g2s, args_retroformer, args_g2s,
                            vocab, vocab_tokens, device):
    '''This is a batchrized version (for retroformer) of run_ensemble for faster evaluation'''
    reactants, scores = run_retroformer(model_retroformer, args_retroformer, smi_list)
    for i, smi in enumerate(smi_list):
        result = run_g2s(model_g2s, args_g2s, smi, vocab, vocab_tokens, device)
        reactants[i] = result['reactants'] + reactants[i]
        scores[i] = result['scores'] + scores[i]
    
    reactants_final = []
    for i in range(len(smi_list)):
        _reactants_final, _scores_final = [], []
        for reactant, score in zip(reactants[i], scores[i]):
            if reactant not in _reactants_final:
                _reactants_final.append(reactant)
                _scores_final.append(score)
        
        ordering = np.argsort(_scores_final)[::-1][:10]
        _reactants_final = list(np.array(_reactants_final)[ordering])
        reactants_final.append(_reactants_final)
    return reactants_final


if __name__ == '__main__':
    t_start = time()
    smi = 'CCCCCCCCCCCCCCCC(=O)OC(CCCCCCCCCCCCC)CC(=O)NCC(=O)O'
    model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, device = prepare_ensemble()
    print(run_ensemble(smi, model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, device))
    print(f'\033[92mTotal {time() - t_start:.2f} sec elapsed\033[0m')
