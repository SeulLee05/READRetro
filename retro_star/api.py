import torch
import pandas as pd
from retro_star.common import prepare_starting_molecules, \
     prepare_molstar_planner, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from retroformer.translate import prepare_retroformer, run_retroformer
from megan.translate import prepare_megan, run_megan
from utils.ensemble import prepare_ensemble, run_ensemble, prepare_g2s, run_g2s
from retro_star.retriever import Retriever, run_retriever, run_retriever_only, \
    neutralize_atoms, kegg_search, \
    pathRetriever, run_path_retriever, run_both_retriever
from rdkit import Chem


class RSPlanner:
    def __init__(self,
                 cuda=True,
                 beam_size=10,
                 iterations=500,
                 expansion_topk=10,
                 use_value_fn=True,
                 starting_molecules='data/building_block.csv',
                 value_model='retro_star/saved_model/best_epoch_final_4.pt',
                 model_type='ensemble',
                 model_path=None,
                 retrieval_db='data/train_canonicalized.txt',
                 path_retrieval_db='data/pathways.pickle',
                 kegg_mol_db = "data/kegg_neutral_iso_smi.csv",
                 route_topk=10,
                 retrieval=True,
                 path_retrieval=True):
        
        setup_logger()
        device = torch.device('cuda' if cuda else 'cpu')
        starting_mols = prepare_starting_molecules(starting_molecules)

        self.path_retrieve_token = None
        self.kegg_mol_db = None
        self.path_retrieval_db = None

        if model_path is None:
            retroformer_path = 'retroformer/saved_models/biochem.pt'
            g2s_path = 'g2s/saved_models/biochem.pt'
        
        if model_type == 'retroformer':
            if model_path is not None:
                retroformer_path = model_path
            
            model_retroformer, args = prepare_retroformer(cuda, beam_size, expansion_topk,
                                                          path=retroformer_path)
            expansion_handler = lambda x: run_retroformer(model_retroformer, args, x)

        elif model_type == 'g2s':
            if model_path is not None:
                g2s_path = model_path
            
            model_g2s, args_g2s, vocab, vocab_tokens, _ = prepare_g2s(cuda, beam_size, expansion_topk,
                                                                      path=g2s_path)
            expansion_handler = lambda x: run_g2s(model_g2s, args_g2s, x, vocab, vocab_tokens, device)

        elif model_type == 'megan':
            model_megan, action_vocab, base_action_masks = prepare_megan(cuda,
                                                                         path=model_path)
            expansion_handler = lambda x: run_megan(model_megan, action_vocab, base_action_masks, x)

        elif model_type == "retriever_only":
            self.path_retrieve_token = 'keggpath'
            self.kegg_mol_db = pd.read_csv(kegg_mol_db)
            self.path_retrieval_db = pd.read_pickle(path_retrieval_db)
            
            starting_mols.add(self.path_retrieve_token)
            path_retriever = pathRetriever(kegg_mol_db, path_retrieval_db, self.path_retrieve_token)
            retriever = Retriever(retrieval_db)
            expansion_handler = lambda x: run_retriever_only(x, path_retriever, retriever)
        
        elif path_retrieval and retrieval:
            self.path_retrieve_token = 'keggpath'
            self.kegg_mol_db = pd.read_csv(kegg_mol_db)
            self.path_retrieval_db = pd.read_pickle(path_retrieval_db)
            
            model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, _ = \
                prepare_ensemble(cuda, beam_size, expansion_topk)

            starting_mols.add(self.path_retrieve_token)
            path_retriever = pathRetriever(kegg_mol_db, path_retrieval_db, self.path_retrieve_token)
            retriever = Retriever(retrieval_db)
            expansion_handler = lambda x: run_both_retriever(x, path_retriever, retriever, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                                        vocab, vocab_tokens, device, expansion_topk)
        
        elif path_retrieval and not retrieval:
            self.path_retrieve_token = 'keggpath'
            self.kegg_mol_db = pd.read_csv(kegg_mol_db)
            self.path_retrieval_db = pd.read_pickle(path_retrieval_db)
            
            model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, _ = \
                prepare_ensemble(cuda, beam_size, expansion_topk)
            
            starting_mols.add(self.path_retrieve_token)
            path_retriever = pathRetriever(kegg_mol_db, path_retrieval_db, self.path_retrieve_token)
            expansion_handler = lambda x: run_path_retriever(x, path_retriever, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                                        vocab, vocab_tokens, device, expansion_topk)
        
        elif not path_retrieval and retrieval:  # READRetro
            if model_path is not None:
                retroformer_path, g2s_path = model_path.split(',')

            model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, _ = \
                prepare_ensemble(cuda, beam_size, expansion_topk,
                                 retroformer_path=retroformer_path,
                                 g2s_path=g2s_path)
            
            retriever = Retriever(retrieval_db)
            expansion_handler = lambda x: run_retriever(x, retriever, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                                        vocab, vocab_tokens, device, expansion_topk)
        
        else:   # READRetro w/o reaction retriever
            model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, _ = \
                prepare_ensemble(cuda, beam_size, expansion_topk,
                                 retroformer_path=retroformer_path,
                                 g2s_path=g2s_path)
            
            expansion_handler = lambda x: run_ensemble(x, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                                       vocab, vocab_tokens, device, expansion_topk)

        self.top_k = route_topk

        if use_value_fn:
            model = ValueMLP(
                n_layers=1,
                fp_dim=2048,
                latent_dim=128,
                dropout_rate=0.1,
                device=device
            ).to(device)
            model.load_state_dict(torch.load(value_model, map_location=device))
            model.eval()

            def value_fn(mol, retrieved):
                if retrieved: return 0.
                # import pdb; pdb.set_trace()
                fp = smiles_to_fp(mol, fp_dim=2048).reshape(1, -1)
                fp = torch.FloatTensor(fp).to(device)
                v = model(fp).item()
                return v
        else:
            value_fn = lambda x, r: 0.

        self.plan_handle = prepare_molstar_planner(
            expansion_handler=expansion_handler,
            value_fn=value_fn,
            starting_mols=starting_mols,
            iterations=iterations
        )

    def __keggpath_find(self,routes,token,mol_db,path_db,top_k):
        modi_list = []
        for route in routes[:top_k]:
            r = route.split(">")
            token_position = [i for i,j in enumerate(r) if token in j]
            for pos in token_position:
                cid, _ = kegg_search(neutralize_atoms(r[pos-2].split("|")[-1]),mol_db)
                target_maps = path_db["Map"][path_db['Pathways'].apply(lambda x: any(cid in sublist for sublist in x))].to_list()
                map = target_maps[0]  # check a representation method
                r[pos] = r[pos].replace(token,f'{token}=kegg.jp/pathway/{map}+{cid}')
                if target_maps == []:  # not the case
                    modi_list.append(route)

            modi_route = '>'.join(r)
            modi_list.append(modi_route)
        return modi_list
    
    def plan(self, target_mol):
        try:
            target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(target_mol))
            succ, msg = self.plan_handle(target_mol)    # the result of model
            
            if succ:
                routes_list = []
                for route in msg:
                    routes_list.append(route.serialize())
                modi_routes_list = self.__keggpath_find(routes_list,self.path_retrieve_token,self.kegg_mol_db,self.path_retrieval_db,self.top_k)
                return modi_routes_list
            
            elif target_mol != Chem.MolToSmiles(Chem.MolFromSmiles(target_mol),isomericSmiles=False):
                target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(target_mol),isomericSmiles=False)
                succ, msg = self.plan_handle(target_mol)    # the result of model
                if succ:
                    routes_list = []
                    for route in msg:
                        routes_list.append(route.serialize())
                    modi_routes_list = self.__keggpath_find(routes_list,self.path_retrieve_token,self.kegg_mol_db,self.path_retrieval_db,self.top_k)
                    return modi_routes_list
                else:
                    return None
            else:
                return None
        except:
            return None