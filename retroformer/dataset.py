import os
import pickle
import numpy as np
from rdkit import Chem

from torch.utils.data import Dataset
from retroformer.utils.smiles_utils import smi_tokenizer, clear_map_number, SmilesGraph


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean) ** 2 / (2 * standard_deviation ** 2))


class RetroDataset(Dataset):
    def __init__(self, smi, mode, data_folder='./data', vocab_file='./vocab_share.pk', augment=False):
        self.data_folder = data_folder

        assert mode in ['train', 'test', 'val']
        self.BondTypes = ['NONE', 'AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE']
        self.bondtoi = {bond: i for i, bond in enumerate(self.BondTypes)}

        self.mode = mode
        self.augment = augment

        with open(vocab_file, 'rb') as f:
            self.src_itos, self.tgt_itos = pickle.load(f)
        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}

        self.processed = self.build_processed_data(smi)

        self.factor_func = lambda x: (1 + gaussian(x, 5.55391565, 0.27170542, 1.20071279)) # pre-computed

    def build_processed_data(self, smi):
        result = self.parse_smi(smi)
        if result is not None:
            src, src_graph = result
            return {'src': src, 'src_graph': src_graph, 'raw_product': smi}
        else:
            print('Warning. Process Failed.')
            return None

    def parse_smi(self, prod):
        cano_prod = clear_map_number(prod)
        if Chem.MolFromSmiles(cano_prod) is None:
            return None

        # Get the smiles graph
        smiles_graph = SmilesGraph(cano_prod)

        # Prepare model inputs
        src_token = smi_tokenizer(cano_prod)
        src_token = ['<UNK>'] + src_token
        src_token = [self.src_stoi.get(st, self.src_stoi['<unk>']) for st in src_token]

        return src_token, smiles_graph

    def reconstruct_smi(self, tokens, src=True, raw=False):
        if src:
            if raw:
                return [self.src_itos[t] for t in tokens]
            else:
                return [self.src_itos[t] for t in tokens if t != 1]
        else:
            if raw:
                return [self.tgt_itos[t] for t in tokens]
            else:
                return [self.tgt_itos[t] for t in tokens if t not in [1, 2, 3]]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        src, src_graph = self.processed['src'], self.processed['src_graph']
        src[0] = self.src_stoi['<UNK>']

        return src, src_graph
