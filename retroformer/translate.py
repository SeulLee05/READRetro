import math
from time import time
import numpy as np
import pickle
from easydict import EasyDict

import os
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore")

import sys
root = os.path.abspath('.')
if root not in sys.path:
    sys.path.insert(0, root)

from retroformer.utils.smiles_utils import *
from retroformer.utils.translate_utils import translate_batch_stepwise
from retroformer.utils.build_utils import build_model, load_checkpoint, build_batch

from rdkit import Chem
    

def filter_invalid(generations, scores):
    filtered_generations, filtered_scores = [], []
    for generation, score in zip(generations, scores):
        mol = Chem.MolFromSmiles(generation)
        if mol is None:
            continue
        filtered_generations.append(Chem.MolToSmiles(mol))
        filtered_scores.append(score)
    
    return filtered_generations, filtered_scores


def filter_single_atom(generations):
    filtered_generations = []
    for generation in generations:
        filtered_generations.append('.'.join([smi for smi in generation.split('.') if len(smi) > 1]))
    
    return filtered_generations


def translate(args, batch, models, dataset, singlestep=False):
    generations, scores = [], []
    invalid_token_indices = [dataset.tgt_stoi['<RX_{}>'.format(i)] for i in range(1, 11)]
    invalid_token_indices += [dataset.tgt_stoi['<UNK>'], dataset.tgt_stoi['<unk>']]
    
    generation_dict, counter = defaultdict(int), defaultdict(int)
    for model in models:
        pred_tokens, pred_scores, predicts = \
            translate_batch_stepwise(model, batch, invalid_token_indices=invalid_token_indices,
                                    beam_size=args.beam_size,
                                    T=10, alpha_atom=-1, alpha_bond=-1,
                                    beta=0.5, percent_aa=40, percent_ab=40, k=3,
                                    use_template=False,
                                    factor_func=dataset.factor_func,
                                    reconstruct_func=dataset.reconstruct_smi)
        if not singlestep:
            assert len(predicts) == 1

        original_beam_size = pred_tokens.shape[1]
        current_i = 0
        for predict in predicts:
            remain = original_beam_size
            beam_size = math.ceil(original_beam_size / len(predict))

            hypo_i, hypo_scores_i = [], []
            for j, (rc, rc_score) in enumerate(predict):
                pred_token = pred_tokens[current_i + j]

                sub_hypo_candidates, sub_score_candidates = [], []
                for k in range(pred_token.shape[0]):
                    hypo_smiles_k = ''.join(dataset.reconstruct_smi(pred_token[k], src=False))
                    hypo_lens_k = len(smi_tokenizer(hypo_smiles_k))
                    hypo_scores_k = pred_scores[current_i + j][k].cpu().numpy() / hypo_lens_k + rc_score / 10

                    if hypo_smiles_k not in hypo_i:  # only select unique entries
                        sub_hypo_candidates.append(hypo_smiles_k)
                        sub_score_candidates.append(hypo_scores_k)

                ordering = np.argsort(sub_score_candidates)[::-1]
                sub_hypo_candidates = list(np.array(sub_hypo_candidates)[ordering])[:min(beam_size, remain)]
                sub_score_candidates = list(np.array(sub_score_candidates)[ordering])[:min(beam_size, remain)]

                hypo_i += sub_hypo_candidates
                hypo_scores_i += sub_score_candidates

                remain -= beam_size

            current_i += len(predict)
            ordering = np.argsort(hypo_scores_i)[::-1]
            generations.append(np.array(hypo_i)[ordering][:10])
            scores.append(np.array(hypo_scores_i)[ordering][:10])
        
        if singlestep:
            generations = [list(g) for g in generations]
            scores = [list(s) for s in scores]
            for i in range(len(scores)):
                if len(scores[i]) > 0:
                    scores[i] = list(np.exp(np.array(scores[i])) / np.sum(np.exp(np.array(scores[i]))))     # softmax
            return generations, scores

        _generations = list(np.array(hypo_i)[ordering])
        _scores = list(np.array(hypo_scores_i)[ordering])

        _generations, _scores = filter_invalid(_generations, _scores)
        _generations = filter_single_atom(_generations)

        for i, (g, s) in enumerate(zip(_generations, _scores)):
            generation_dict[g] += s
            counter[g] += 1

    for g, cnt in counter.items():
        generation_dict[g] /= cnt
    
    generations = list(generation_dict.keys())
    scores = list(generation_dict.values())
    
    ordering = np.argsort(scores)[::-1][:args.expansion_topk]
    generations = list(np.array(generations)[ordering])
    scores = list(np.array(scores)[ordering])

    if len(scores) > 0:
        scores = list(np.exp(np.array(scores)) / np.sum(np.exp(np.array(scores))))   # softmax

    return {'reactants': generations, 'scores': scores}


def prepare_retroformer(cuda=True, beam_size=10, expansion_topk=10,
                        path='retroformer/saved_models/biochem.pt',
                        vocab_path='retroformer/saved_models/vocab_share.pk'):
    args = EasyDict()
    args.device = 'cuda' if cuda else 'cpu'
    args.beam_size = beam_size
    args.expansion_topk = expansion_topk
    args.vocab = vocab_path
    args.checkpoint = path
    
    with open(args.vocab, 'rb') as f:
        src_itos, tgt_itos = pickle.load(f)

    # ensemble
    args.checkpoint = args.checkpoint.split(',')
    models = [build_model(args, src_itos, tgt_itos) for _ in range(len(args.checkpoint))]
    models = load_checkpoint(args, models)
    return models, args


def run_retroformer(model, args, smi):
    singlestep = True
    if type(smi) == str:
        smi = [smi]
        singlestep = False
    batch, dataset = build_batch(args, smi)
    result = translate(args, batch, model, dataset, singlestep)
    if singlestep:
        return result
    result['templates'] = [None for _ in range(len(result['scores']))]
    result['retrieved'] = [False for _ in range(len(result['scores']))]
    return result


if __name__ == '__main__':
    t_start = time()
    model, args = prepare_retroformer()
    # smi = 'C=CC(C)(C)[C@@]12C[C@H]3c4nc5ccccc5c(=O)n4[C@H](C)C(=O)N3C1N(C(C)=O)c1ccccc12'
    smi = ['C=CC(C)(C)[C@@]12C[C@H]3c4nc5ccccc5c(=O)n4[C@H](C)C(=O)N3C1N(C(C)=O)c1ccccc12', 'N[C@@H](CNC(=O)C(=O)O)C(=O)O']
    print(run_retroformer(model, args, smi))
    print(f'\033[92mTotal {time() - t_start:.2f} sec elapsed\033[0m')
