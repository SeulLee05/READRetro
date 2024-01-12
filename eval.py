import argparse


def dict_update(tdict, tkey, tval):
    if tkey in tdict:
        pos = len(tdict[tkey])
        tdict[tkey].update({pos: tval})
    else:
        tdict[tkey] = {0: tval}


def get_pred_dict(pth, mol2class, product_class='all'):
    mol_dict= {}
    with open(pth, 'r') as f:
        lines = [i.strip('\n') for i in f.readlines()]
    for line in lines:
        path = line.split(' ')[-1]
        if path in {'None', 'Error'}:
            continue
        product = path.split('>')[0]
        if product_class == 'all' or mol2class[product] == product_class:
        # if mol2class[product] != 'AA-MA':
            dict_update(mol_dict, product, path)
    return mol_dict


def get_ground_truth_dict(pth):
    '''
    :param pth: the path of ground truth file(.txt)
    :return: a dict, format: {
                    smiles1: {0: ground truth path1; 1: ground truth path2; ...}
                    smiles2: {0: ground truth path1; 1: ground truth path2; ...}
                }
    '''
    ground_truth_dict = {}
    with open(pth, 'r') as f:
        l = [i.strip('\n') for i in f.readlines()]
    
    for line in l:
        ele = line.split('\t')
        ele = [i for i in ele if len(i) != 0]
        target = ele[3]
        dict_update(ground_truth_dict, target, '|'.join(ele[3:]))
    return ground_truth_dict


lengths = []
def get_lst_from_pred_dict(s):  
    '''
    :param s: 'smiles1>score1>smiles2|smiles2>score2>smiles3|smiles3>score3>smiles4...'
    :return: [smiles1, smiles2, smiles3, ...]
    '''
    lst = []
    rea = s.split('|')
    lengths.append(len(rea))
    for i in range(len(rea)):
        line = rea[i].split('>')
        if i == 0:
            lst.append(line[0])
        lst.append(line[-1])
    return lst


def get_lst_from_gt_dict(s):
    '''
    :param s: 'smiles1|smiles2|smiles3...'
    :return: [smiles1, smiles2, smiles3, ...]
    '''
    l = s.split('|')
    mols = []
    for i in l:
        try:
            mols.append(i)
        except:
            print(f"wrong reaction: {i}")
    return mols


def blockHit(ori_path, pred_path): # if the results structure of building block prediction is like a normal path (block => i1 => i2 ... in => product)
    ori_bb = ori_path[-1]
    pred_bb = pred_path[-1]
    return ori_bb == pred_bb


def pathHit(ori_path, pred_path):
    ori_set = set(ori_path)
    pred_set = set(pred_path)
    intersect = len(ori_set & pred_set)
    num = 0
    if len(ori_path) == len(pred_path) and intersect == len(ori_path):
        num = 1
    return num, intersect - 1


class MultiPredRes:
    def __init__(self, pred_p, truth_p, testSetLength):
        self.PredictionName = pred_p
        self.GtName = truth_p
        self.successPath = len(pred_p)
        self.blockHit = 0
        self.pathHit = 0
        self.globalMax_intersect = 0
        self.globalMax_gt = 0
        self.testSetLen = testSetLength
        self.result_dict = {}

    def resultShow(self, toFile=""):
        s = ""
        s += f"Success rate:\t\t\t{self.successPath / self.testSetLen*100:.4f}%\n"
        s += f"Hit rate of building blocks:\t{self.blockHit / self.testSetLen*100:.4f}%\n"
        s += f"Hit rate of pathways:\t\t{self.pathHit / self.testSetLen*100:.4f}%"
        
        if toFile:
            with open(toFile, "w") as f:
                f.write(s)
        print(s)
        return s


def multiVal(pred_path, product_class='all'):
    with open('data/test.txt') as f:
        mol2class = f.readlines()
    mol2class = [l.strip().split('\t') for l in mol2class]
    mol2class = {l[2]: l[1] for l in mol2class}

    products = set()
    for product, _class in mol2class.items():
        if product_class == 'all' or _class == product_class:
        # if mol2class[product] != 'AA-MA':
            products.add(product)
    testSetNum = len(products)
    print(f'Number of test molecules:\t{testSetNum}')

    pred_dict = get_pred_dict(pred_path, mol2class, product_class)
    # num = 0
    # for res in pred_dict.values():
    #     num += np.clip(len(res), None, 5)
    # print(num / len(pred_dict))
    # import pdb; pdb.set_trace()

    ground_truth_dict = get_ground_truth_dict('data/test_gt.txt')
    # print(pred_path)
    mRes = MultiPredRes(pred_dict, ground_truth_dict, testSetNum)

    for smiles, v_dict in pred_dict.items():
        if v_dict is None:
            continue
        try:
            gt_dict = ground_truth_dict[smiles]
        except:
            import pdb; pdb.set_trace()
            print("fail")
        building_block_same = False
        path_same = False
        max_intersect = 0
        max_pred = ''
        max_gt = ''
        for routes_score in v_dict.values():
            for item in gt_dict.values():
                gt_lst = get_lst_from_gt_dict(item)
                pred_lst = get_lst_from_pred_dict(routes_score)

                bb_same = blockHit(gt_lst, pred_lst)
                if bb_same:
                    building_block_same = True

                num, intersect = pathHit(gt_lst, pred_lst)
                if num:
                    path_same = True

                if intersect > max_intersect:
                    max_intersect = intersect
                    max_pred = pred_lst
                    max_gt = gt_lst
        mRes.globalMax_intersect = max(mRes.globalMax_intersect, max_intersect)

        if building_block_same:
            mRes.blockHit += 1
        if path_same:
            mRes.pathHit += 1
        mRes.result_dict[smiles] = {'max_intersect': max_intersect, 'max_pred': '>>'.join(max_pred),
                                    'max_gt': '>>'.join(max_gt)}
    mRes.resultShow()


parser = argparse.ArgumentParser()
parser.add_argument('save_file', type=str)
parser.add_argument('-c', '--product_class', type=str, default='all',
                    choices=['all', 'Amino', 'Complex', 'Cinnamic', 'MVA/MEP', 'AA-MA'])
args = parser.parse_args()

# treating alias
if args.product_class == 'Amino':
    args.product_class = 'Amino acid'
elif args.product_class == 'Cinnamic':
    args.product_class = 'Cinnamic/Shikimic acid'


multiVal(args.save_file, args.product_class)
