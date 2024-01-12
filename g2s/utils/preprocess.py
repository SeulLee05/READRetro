import numpy as np
from typing import Dict, List, Tuple
from g2s.utils.data_utils import get_graph_features_from_smi, load_vocab, tokenize_smiles, G2SDataset_single


def get_token_ids(tokens: list, vocab: Dict[str, int], max_len: int) -> Tuple[List, int]:
    token_ids = []
    token_ids.extend([vocab[token] for token in tokens])
    token_ids = token_ids[:max_len-1]
    token_ids.append(vocab["_EOS"])

    lens = len(token_ids)
    while len(token_ids) < max_len:
        token_ids.append(vocab["_PAD"])

    return token_ids, lens


def get_seq_features_from_line_single(smi, vocab, max_src_len=1024):
    src_tokens = smi.strip().split()
    if not src_tokens:
        src_tokens = ["C", "C"]             # hardcode to ignore
    src_token_ids, src_lens = get_token_ids(src_tokens, vocab, max_len=max_src_len)
    src_token_ids = np.array(src_token_ids, dtype=np.int32)
    return src_token_ids, src_lens


def preprocess_smi(smi, vocab):
    smi = tokenize_smiles(''.join(smi.strip().split()))
    src_token_ids, src_lens = get_seq_features_from_line_single(smi, vocab)

    src_token_ids = np.stack([src_token_ids], axis=0)
    src_lens = np.array([src_lens], dtype=np.int32)

    a_scopes, a_scopes_lens, b_scopes, b_scopes_lens, a_features, a_features_lens, \
        b_features, b_features_lens, a_graphs, b_graphs = get_graph_features_from_smi((0, "".join(smi.split()), False))

    a_scopes = np.concatenate([a_scopes], axis=0)
    b_scopes = np.concatenate([b_scopes], axis=0)
    a_features = np.concatenate([a_features], axis=0)
    b_features = np.concatenate([b_features], axis=0)
    a_graphs = np.concatenate([a_graphs], axis=0)
    b_graphs = np.concatenate([b_graphs], axis=0)

    a_scopes_lens = np.array([a_scopes_lens], dtype=np.int32)
    b_scopes_lens = np.array([b_scopes_lens], dtype=np.int32)
    a_features_lens = np.array([a_features_lens], dtype=np.int32)
    b_features_lens = np.array([b_features_lens], dtype=np.int32)

    feat = {'src_token_ids': src_token_ids, 'src_lens': src_lens, 'a_scopes': a_scopes, 'b_scopes': b_scopes,
            'a_features': a_features, 'b_features': b_features, 'a_graphs': a_graphs, 'b_graphs': b_graphs,
            'a_scopes_lens': a_scopes_lens, 'b_scopes_lens': b_scopes_lens,
            'a_features_lens': a_features_lens, 'b_features_lens': b_features_lens}
    return feat


def build_batch(smi, vocab):
    feat = preprocess_smi(smi, vocab)
    dataset = G2SDataset_single(feat, vocab)
    return dataset[0]


if __name__ == "__main__":
    vocab = load_vocab('data/vocab_smiles.txt')
    batch = build_batch('C C ( = O ) N / C = C ( / C C ( = O ) O ) C ( = O ) O', vocab)
    import pdb; pdb.set_trace()
