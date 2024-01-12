from retro_star.api import RSPlanner
from time import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('product',                      type=str)
parser.add_argument('-b', '--blocks',               type=str, default='data/building_block.csv')
parser.add_argument('-i', '--iterations',           type=int, default=20)
parser.add_argument('-e', '--exp_topk',             type=int, default=10)
parser.add_argument('-k', '--route_topk',           type=int, default=10)
parser.add_argument('-s', '--beam_size',            type=int, default=10)
parser.add_argument('-m', '--model_type',           type=str, default='ensemble', choices=['ensemble', 'retroformer', 'g2s', 'retriever_only', 'megan'])
parser.add_argument('-mp', '--model_path',          type=str, default=None)
parser.add_argument('-r', '--retrieval',            type=str, default='true', choices=['true', 'false'])
parser.add_argument('-pr', '--path_retrieval',      type=str, default='false', choices=['true', 'false'])
parser.add_argument('-d', '--retrieval_db',         type=str, default='data/train_canonicalized.txt')
parser.add_argument('-pd', '--path_retrieval_db',   type=str, default='data/pathways.pickle')
parser.add_argument('-c', '--device',               type=str, default='cuda', choices=['cuda', 'cpu'])
args = parser.parse_args()

t_start = time()

planner = RSPlanner(
    cuda=args.device=='cuda',
    iterations=args.iterations,
    expansion_topk=args.exp_topk,
    route_topk=args.route_topk,
    beam_size=args.beam_size,
    model_type=args.model_type,
    model_path=args.model_path,
    retrieval=args.retrieval=='true',
    retrieval_db=args.retrieval_db,
    path_retrieval=args.path_retrieval=='true',
    path_retrieval_db=args.path_retrieval_db,
    starting_molecules=args.blocks
)

result = planner.plan(args.product)

if result is None:
    print('None')
else:
    for i, route in enumerate(result):
        print(f'{i} {route}')

print(f'\033[92mTotal {time() - t_start:.2f} sec elapsed\033[0m')
