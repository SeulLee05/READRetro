import os
from time import time
import numpy as np
import argparse
import pandas as pd
import multiprocessing as mp
from utils.multithread_utils import *

mp.set_start_method('forkserver', force=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--products',             type=str, default='data/test.txt')
    parser.add_argument('-f', '--save_file',            type=str, default='result/debug.txt')
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
    parser.add_argument('-n', '--num_threads',          type=int, default=10)
    args = parser.parse_args()

    t_start = time()

    if not os.path.exists(os.path.dirname(args.save_file)):
        os.makedirs(os.path.dirname(args.save_file))

    df = pd.read_csv(args.products, header=None, sep='\t')
    mol_ids, products = df[1], df[2]
    
    idx_split = np.array_split(range(len(products)), args.num_threads)
    idx_split = [(i[0], i[-1] + 1) for i in idx_split]
    print(idx_split)

    manager = mp.Manager()
    q = manager.Queue()
    file_pool = mp.Pool(1)
    file_pool.apply_async(listener, (q, args.save_file))

    pool = mp.Pool(args.num_threads)
    jobs = []
    for i in range(args.num_threads):
        start, end = idx_split[i]
        job = pool.apply_async(worker, (q, i, args, mol_ids[start:end], products[start:end]))
        jobs.append(job)

    for job in jobs:
        job.get()

    q.put('#done#')
    pool.close()
    pool.join()

    print(f'\033[92mTotal {time() - t_start:.2f} sec elapsed\033[0m')
