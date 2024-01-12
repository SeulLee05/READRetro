from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import torch
from retroformer.models.model import RetroModel
from retroformer.dataset import RetroDataset


def load_checkpoint(args, models):
    for checkpoint_path, model in zip(args.checkpoint, models):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.to(args.device)
    return models


def build_model(args, vocab_itos_src, vocab_itos_tgt):
    src_pad_idx = np.argwhere(np.array(vocab_itos_src) == '<pad>')[0][0]
    tgt_pad_idx = np.argwhere(np.array(vocab_itos_tgt) == '<pad>')[0][0]

    model = RetroModel(
        encoder_num_layers=8,
        decoder_num_layers=8,
        d_model=256, heads=8, d_ff=2048, dropout=0.1,
        vocab_size_src=len(vocab_itos_src), vocab_size_tgt=len(vocab_itos_tgt),
        shared_vocab=True, shared_encoder=False,
        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx)

    return model.to(args.device)


def build_batch(args, smi_list):
    dataset_list = []
    for smi in smi_list:
        dataset = RetroDataset(smi, mode='test', vocab_file=args.vocab)
        dataset_list.append(dataset[0])
    src_pad = dataset.src_stoi['<pad>']
    batch = collate_fn(dataset_list, src_pad)
    return batch, dataset


def collate_fn(data, src_pad, device='cuda'):
    """Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    """
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    src, src_graph = zip(*data)
    max_src_length = max([len(s) for s in src])

    anchor = torch.zeros([], device=device)

    # Graph structure with edge attributes
    new_bond_matrix = anchor.new_zeros((len(data), max_src_length, max_src_length, 7), dtype=torch.long)

    # Pad_sequence
    new_src = anchor.new_full((max_src_length, len(data)), src_pad, dtype=torch.long)

    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.LongTensor(src[i])

        full_adj_matrix = torch.from_numpy(src_graph[i].full_adjacency_tensor)
        new_bond_matrix[i, 1:full_adj_matrix.shape[0]+1, 1:full_adj_matrix.shape[1]+1] = full_adj_matrix

    return new_src, (new_bond_matrix, src_graph)
