import math
import argparse
from tqdm import tqdm
from rdkit import Chem
from retroformer.translate import prepare_retroformer, run_retroformer
from g2s.translate import prepare_g2s, run_g2s
from utils.ensemble import prepare_ensemble, run_ensemble_singlestep


def remove_chiral(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def cal_acc(preds, target, n_best):
    correct_cnt = {'top1': 0, 'top3': 0, 'top5': 0, 'top10': 0}
    for i, tgt_smi in enumerate(target):
        pred_list = preds[i][:n_best]
        if tgt_smi in pred_list[:1]:
            correct_cnt['top1'] += 1
        if tgt_smi in pred_list[:3]:
            correct_cnt['top3'] += 1
        if tgt_smi in pred_list[:5]:
            correct_cnt['top5'] += 1
        if tgt_smi in pred_list[:10]:
            correct_cnt['top10'] += 1
    acc_dict = {key: value / len(target) for key, value in correct_cnt.items()}
    return acc_dict


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type',       type=str, default='ensemble', choices=['ensemble', 'retroformer', 'g2s'])
parser.add_argument('-p', '--model_path',       type=str, default=None)
parser.add_argument('-v', '--model_vocab_path', type=str, default=None)
parser.add_argument('-b', '--batch_size',       type=int, default=32)
parser.add_argument('-s', '--beam_size',        type=int, default=10)
parser.add_argument('-d', '--device',           type=str, default='cuda', choices=['cuda', 'cpu'])
args = parser.parse_args()

if args.model_path is None:
    retroformer_path = 'retroformer/saved_models/biochem.pt'
    g2s_path = 'g2s/saved_models/biochem.pt'

if args.model_vocab_path is None:
    retroformer_vocab_path = 'retroformer/saved_models/vocab_share.pk'
    g2s_vocab_path = 'g2s/saved_models/vocab_smiles.txt'

if args.model_type == 'ensemble':
    if args.model_path is not None:
        retroformer_path, g2s_path = args.model_path.split(',')

    if args.model_vocab_path is not None:
        retroformer_vocab_path, g2s_vocab_path = args.model_vocab_path.split(',')

    model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, device = \
        prepare_ensemble(args.device == 'cuda', args.beam_size,
                         retroformer_path=retroformer_path,
                         retroformer_vocab_path=retroformer_vocab_path,
                         g2s_path=g2s_path,
                         g2s_vocab_path=g2s_vocab_path)
elif args.model_type == 'retroformer':
    if args.model_path is not None:
        retroformer_path = args.model_path
    
    if args.model_vocab_path is not None:
        retroformer_vocab_path = args.model_vocab_path
    
    model_retroformer, args_retroformer = prepare_retroformer(args.device == 'cuda', args.beam_size,
                                                              path=retroformer_path,
                                                              vocab_path=retroformer_vocab_path)
else:
    if args.model_path is not None:
        g2s_path = args.model_path
    
    if args.model_vocab_path is not None:
        g2s_vocab_path = args.model_vocab_path
    
    model_g2s, args_g2s, vocab, vocab_tokens, device = prepare_g2s(args.device == 'cuda', args.beam_size,
                                                                   path=g2s_path,
                                                                   vocab_path=g2s_vocab_path)

with open('data/test_canonicalized_single.txt', 'r') as f:
    reactions = f.readlines()
tgt = [r.strip().split('>>')[0] for r in reactions]
src = [r.strip().split('>>')[1] for r in reactions]
tgt = [remove_chiral(t) for t in tgt]

preds = []
if args.model_type == 'ensemble':
    for i in tqdm(range(math.ceil(len(src) / args.batch_size))):
        smi_list = src[i * args.batch_size : (i + 1) * args.batch_size]
        preds.extend(run_ensemble_singlestep(smi_list, model_retroformer, model_g2s,
                                             args_retroformer, args_g2s,
                                             vocab, vocab_tokens, device))
elif args.model_type == 'retroformer':
    for i in tqdm(range(math.ceil(len(src) / args.batch_size))):
        smi_list = src[i * args.batch_size : (i + 1) * args.batch_size]
        preds.extend(run_retroformer(model_retroformer, args_retroformer, smi_list)[0])
else:
    for smi in tqdm(src):
        preds.append(run_g2s(model_g2s, args_g2s, smi, vocab, vocab_tokens, device)['reactants'])

for pred in preds:
    for i in range(len(pred)):
        pred[i] = remove_chiral(pred[i])

result = cal_acc(preds, tgt, n_best=10)
print(result)
