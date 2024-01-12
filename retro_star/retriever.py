from collections import defaultdict
from utils.ensemble import run_ensemble
import pandas as pd
import numpy as np
from rdkit import Chem

def neutralize_atoms(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        return Chem.MolToSmiles(mol,isomericSmiles=False)
    except:
        return smi

def kegg_search(smi,kegg_df):
    extract = kegg_df[kegg_df['SMILES'] == smi]
    if not len(extract):
        return None, None
    else:
        id = extract["ID"].values[0]
        return id, f"www.kegg.jp/entry/{id}"

def common_find(target_key, dict1, dict2):
    common_indices_list1 = [i for i, x in enumerate(dict1[target_key]) if x in dict2[target_key]]
    common_indices_list2 = [i for i, x in enumerate(dict2[target_key]) if x in dict1[target_key]]
    
    common_dict1, common_dict2 = {}, {}
    new_dict1, new_dict2 = {}, {}
    
    for key in dict1:
        values = dict1[key]
        new_dict1[key] = [value for i, value in enumerate(values) if i not in common_indices_list1]
        common_dict1[key] = [value for i, value in enumerate(values) if i in common_indices_list1]

    for key in dict2:
        values = dict2[key]
        new_dict2[key] = [value for i, value in enumerate(values) if i not in common_indices_list2]
        common_dict2[key] = [value for i, value in enumerate(values) if i in common_indices_list2]
    merged_dict = {}

    for key in dict1.keys():
        merged_dict[key] = new_dict1[key] + new_dict2[key]
    return common_dict1, common_dict2, merged_dict

class Retriever:
    def __init__(self, data):
        with open(data) as f:
            reactions = f.readlines()

        self.graph = defaultdict(list)
        for reaction in reactions:
            reactant, product = reaction.strip().split('>>')
            self.graph[product].append(reactant)
    
    def retrieve(self, product):
        if product in self.graph:
            return self.graph[product]

class pathRetriever:
    def __init__(self, kegg_mol_data, pathway_data, token):
        kegg_mol = pd.read_csv(kegg_mol_data)
        pathways = pd.read_pickle(pathway_data)
        compounds = set([i for sublist in pathways["Compounds"] for i in sublist])
        self.compounds = kegg_mol[kegg_mol.apply(lambda row: row['ID'] in compounds, axis=1)]
        self.special_token = token
        
    def retrieve(self, product):
        id, _ = kegg_search(neutralize_atoms(product),self.compounds)
        return [self.special_token] if id != None else []    # if some compounds do not contain in pathways although in compounds, change the condition 


def run_path_retriever(product, path_retriever, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                  vocab, vocab_tokens, device, expansion_topk):
    isPathknown = path_retriever.retrieve(product)
    if isPathknown:
        res_dict = {'reactants': isPathknown}
        res_dict['scores'] = [1. for _ in res_dict['reactants']]
        res_dict['retrieved'] = [True for _ in res_dict['reactants']]
        res_dict['templates'] = [None for _ in res_dict['scores']]
    else:
        res_dict = run_ensemble(product, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                vocab, vocab_tokens, device, expansion_topk)
    return res_dict


def run_retriever(product, retriever, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                  vocab, vocab_tokens, device, expansion_topk):
    reactant_list = retriever.retrieve(product)
    if reactant_list is not None:
        res_dict = {'reactants': reactant_list}
        res_dict['scores'] = [1. for _ in res_dict['reactants']]
        res_dict['retrieved'] = [True for _ in res_dict['reactants']]
        _res_dict = run_ensemble(product, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                 vocab, vocab_tokens, device, expansion_topk)

        res_dict['reactants'].extend(_res_dict['reactants'])
        res_dict['scores'].extend(_res_dict['scores'])
        res_dict['retrieved'].extend(_res_dict['retrieved']) 
        res_dict['templates'] = [None for _ in res_dict['scores']]
        
    else:
        res_dict = run_ensemble(product, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                vocab, vocab_tokens, device, expansion_topk)
    return res_dict

def run_both_retriever(product, path_retriever, retriever, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                  vocab, vocab_tokens, device, expansion_topk):
    isPathknown = path_retriever.retrieve(product)
    if isPathknown:
        res_dict = {'reactants': isPathknown}
        res_dict['scores'] = [1. for _ in res_dict['reactants']]
        res_dict['retrieved'] = [True for _ in res_dict['reactants']]
        res_dict['templates'] = [None for _ in res_dict['scores']]
        return res_dict
        
    reactant_list = retriever.retrieve(product)
    if reactant_list is not None:
        res_dict = {'reactants': reactant_list}
        res_dict['scores'] = [1. for _ in res_dict['reactants']]
        res_dict['retrieved'] = [True for _ in res_dict['reactants']]
        _res_dict = run_ensemble(product, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                 vocab, vocab_tokens, device, expansion_topk)
        
        res_dict['reactants'].extend(_res_dict['reactants'])
        res_dict['scores'].extend(_res_dict['scores'])
        res_dict['retrieved'].extend(_res_dict['retrieved'])
        res_dict['templates'] = [None for _ in res_dict['scores']]
    else:
        res_dict = run_ensemble(product, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                vocab, vocab_tokens, device, expansion_topk)
    return res_dict

def run_retriever_only(product, path_retriever, retriever):
    isPathknown = path_retriever.retrieve(product)
    if isPathknown:
        res_dict = {'reactants': isPathknown}
        res_dict['scores'] = [1. for _ in res_dict['reactants']]
        res_dict['retrieved'] = [True for _ in res_dict['reactants']]
        res_dict['templates'] = [None for _ in res_dict['scores']]
        return res_dict
    
    reactant_list = retriever.retrieve(product)
    if reactant_list is not None:
        res_dict = {'reactants': reactant_list}
        res_dict['scores'] = [1. for _ in res_dict['reactants']]
        res_dict['retrieved'] = [True for _ in res_dict['reactants']]
        res_dict['templates'] = [None for _ in res_dict['scores']]
    else:
        return {'reactants':[],'scores':[],'retrieved':[],'templates':[]}
    return res_dict
