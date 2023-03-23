# step3:
# based on pdbs with completed side chains, generate pytorch DataLoader for finetuning
import json
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import copy
from random import sample


# this class is used to define the batch generation role of the DataLoader
# data.Sampler: Base class for all Samplers.
# Every Sampler subclass has to provide an __iter__() method, providing a way to iterate over indices of dataset elements, and a __len__() method that returns the length of the returned iterators.
class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.

    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.

    :param node_counts: array of node counts in the dataset to sample from
    :param max_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''

    def __init__(self, node_counts, max_nodes=3000, shuffle=True):

        self.node_counts = node_counts
        # [330, 129, 185, 415, 366], a group of values indicating residue number of each protein
        self.idx = [i for i in range(len(node_counts)) if node_counts[i] <= max_nodes]
        self.shuffle = shuffle
        self.max_nodes = max_nodes
        self._form_batches()

    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        # iterate all idx satisfying the max_nodes limitation
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)

            self.batches.append(batch)
        # print(self.batches, len(self.idx))
        # protein ids for each batch based on the max_nodes
        # [5834, 8761, 11919, 6685, 15116, 15389]
        # [10367, 16609, 11349, 17088, 2584, 5260]
        # [6262, 10342, 1881, 480, 6789, 14797, 4270, 16765, 9676]
        # [7429, 7581, 16596, 17935, 11241, 10539, 9014, 6860]
        # [16145, 4291, 11833, 6806, 10629, 15632, 15518]
        # [1174, 13811, 14623, 14408, 7619, 828, 17099, 10342, 14095]

    def __len__(self):
        if not self.batches: self._form_batches()
        return len(self.batches)

    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch  # used in generator for iteration


# this is data source for finetuning, rather the formal pytorch Dataset sent to pytorch DataLoader
# thus here is only for loading basic data without any extra feature calculation, feature calculation is performed in pytorch Dataset
class FinetuningDataset:
    def __init__(self, path, splits_path):
        with open(splits_path) as f:
            # json.load() is used to read the JSON document from file and json.loads() is used to convert the JSON String document into the Python dictionary.
            dataset_splits = json.load(f)
            # print(dataset_splits['train'], dataset_splits['val'])

        train_list, val_list, test_list = dataset_splits['train'], dataset_splits['val'], dataset_splits['test']
        self.train, self.val, self.test = [], [], []

        with open(path) as f:
            lines = f.readlines()

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)

            name = entry['name']
            wt_complex_coords = entry['widetype_complex']['coords'] # re-arange the N-CA-C-O coordinates
            entry['widetype_complex']['coords'] = list(zip(wt_complex_coords['N'], wt_complex_coords['CA'], wt_complex_coords['C'], wt_complex_coords['O']))
            mt_complex_coords = entry['mutation_complex']['coords'] # re-arange the N-CA-C-O coordinates
            entry['mutation_complex']['coords'] = list(zip(mt_complex_coords['N'], mt_complex_coords['CA'], mt_complex_coords['C'], mt_complex_coords['O']))

            # for the downstream sets, the json entries are stored in the order of the original mutation information files
            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)


class FinetuningCVDataset:
    def __init__(self, path, dataset_splits, extra_split = None):

        train_list, val_list = dataset_splits['train'], dataset_splits['val']
        self.train, self.val = [], []
        if extra_split:
            self.train_val = []
            train_val_num = int(len(train_list) * 0.1)
            train_val_list = sample(train_list, train_val_num)
            train_list = sorted(list(set(train_list) - set(train_val_list)))
            train_val_list = sorted(list(train_val_list))

        with open(path) as f:
            lines = f.readlines()

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)

            name = entry['name']
            wt_complex_coords = entry['widetype_complex']['coords'] # re-arange the N-CA-C-O coordinates
            entry['widetype_complex']['coords'] = list(zip(wt_complex_coords['N'], wt_complex_coords['CA'], wt_complex_coords['C'], wt_complex_coords['O']))
            mt_complex_coords = entry['mutation_complex']['coords'] # re-arange the N-CA-C-O coordinates
            entry['mutation_complex']['coords'] = list(zip(mt_complex_coords['N'], mt_complex_coords['CA'], mt_complex_coords['C'], mt_complex_coords['O']))

            # for the downstream sets, the json entries are stored in the order of the original mutation information files
            if not extra_split:
                if name in train_list:
                    self.train.append(entry)
                elif name in val_list:
                    self.val.append(entry)
            else:
                if name in train_list:
                    self.train.append(entry)
                elif name in val_list:
                    self.val.append(entry)
                elif name in train_val_list:
                    self.train_val.append(entry)


# after defining source data class, we need to define relevant pytorch Dataset class to produce required node and edge features
# here we do not need to add noises to protein graphs, but we need to consider how to generate features for widetype and mutation protein graphs at the same time
class FinetuningGraphDataset(data.Dataset):
    def __init__(self, data_list, num_positional_embeddings=16, top_k = 30, num_rbf=16, device='cpu', sidec_chain_normalization = False,
                 whether_spatial_graph = False, add_mut_to_interface = False, whether_CG_feature=False, CG_encoding_type='sincos'):
        super(FinetuningGraphDataset, self).__init__()

        self.data_list = data_list # training/val list that stores each entry in the format of dict
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(''.join(e['widetype_complex']['seq'].split(':'))) for e in data_list] # used for passing residue node number to PyG DataLoader
        self.add_mut_to_interface = add_mut_to_interface # indicate whether to add mutation cr-token to interface cr_token set

        # we can limit the residue number of a protein in pytorch DataLoader (by detecting and removing it through the set threshold)

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.atom_mass = {'C': 0.1621, 'N': 0.1891, 'O': 0.2160, 'S': 0.4328}
        # temp = 12.011 + 14.0067 + 15.9994 + 32.06
        # print(12.011/temp, 14.0067/temp, 15.9994/temp, 32.06/temp)

        self.sidec_chain_normalization = sidec_chain_normalization # for further normalization of side chain coordinates
        self.whether_spatial_graph = whether_spatial_graph
        self.whether_CG_feature = whether_CG_feature
        self.CG_encoding_type = CG_encoding_type # 'sin'/'onehot'/None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        # The __getitem__ function loads and returns a sample from the dataset at the given index idx.
        return self._featurize_as_graph(self.data_list[i], i)

    # *** need to check whether the generated mask and random coordinate perturbation will change in different epochs ***
    # *** need to ensure no operation will influence the original data stored in self.data_list (protein in _featurize_as_graph in originated from self.data_list[i]) ***
    # *** otherwise the data generated for the next batch will be changed, causing errors ***
    def _featurize_as_graph(self, protein, i):
        # name, widetype_complex, mutation_complex, mutation_info, interface, ddg
        # widetype_complex, mutation_complex include: (seq, coords, num_chains, name, sidechain_dict, sasa_dict, res_idx_list, len_complete_aalist)
        name = protein['name'] # name: widetype protein name + serial number in original mutation file
        with torch.no_grad():
            # here we can mark the interface and mutation residues (and can set a range threshold for them to record an area around interface and mutation sites)
            partner = protein['partner']
            wt = protein['widetype_complex']
            mt = protein['mutation_complex']

            wt_features, mt_features = map(self.feature_generator, (wt, mt))

            if self.whether_spatial_graph:
                # return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list, extra_edge_index, extra_edge_s, extra_edge_v
                wt_node_s, wt_node_v, wt_edge_s, wt_edge_v, wt_edge_index, wt_effective_reslist, wt_extra_edge_index, wt_extra_edge_s, wt_extra_edge_v = \
                    wt_features[0], wt_features[1], wt_features[2], wt_features[3], wt_features[4], wt_features[5], wt_features[6], wt_features[7], wt_features[8]
                mt_node_s, mt_node_v, mt_edge_s, mt_edge_v, mt_edge_index, mt_effective_reslist, mt_extra_edge_index, mt_extra_edge_s, mt_extra_edge_v = \
                    mt_features[0], mt_features[1], mt_features[2], mt_features[3], mt_features[4], mt_features[5], mt_features[6], mt_features[7], mt_features[8]
            else:
                wt_node_s, wt_node_v, wt_edge_s, wt_edge_v, wt_edge_index, wt_effective_reslist = wt_features[0], wt_features[1], wt_features[2], wt_features[3], wt_features[4], wt_features[5]
                mt_node_s, mt_node_v, mt_edge_s, mt_edge_v, mt_edge_index, mt_effective_reslist = mt_features[0], mt_features[1], mt_features[2], mt_features[3], mt_features[4], mt_features[5]

            assert wt_node_s.size() == mt_node_s.size() and wt_node_v.size() == mt_node_v.size() and len(wt_effective_reslist) == len(mt_effective_reslist) and\
                   wt_node_s.size(0) == len(wt_effective_reslist), 'widetype and mutation complexes have different residue node feature number and dim'

            # for GeoPPI, the objective for it to calculate mean and max pooling is the mutation sites and interface sites (including mutation sites)
            # thus, what we can provide here are masks for mutation sites and interface sites, and corresponding mean and max pooling operations are put into finetuning models
            mutation_info, interface, ddg = protein['mutation_info'], protein['interface'], protein['ddg']
            # mutation_info: V:F17A, interface: ['H_L_H_37', 'H_L_H_39'], ddg: 0.0
            # print(wt_effective_reslist) # 'H_221', 'H_222', 'H_223', 'H_224', 'L_1', 'L_2', 'L_3', 'L_4'

            # generate mutation and interface site masks
            mutation_crtoken = ['{}_{}'.format(i.split(':')[0], i.split(':')[-1][1:-1]) for i in mutation_info.split(',')]
            mutation_chainid = [i.split(':')[0] for i in mutation_info.split(',')]
            # for screening mutation chain related residues in the interface generated by pymol (the mutation chain is included in the screened pairwise interfaces by pymol)
            # input: 1. pymol generated interface, 2. chain of interests provided by mutation file, 3. mutation chain
            # output: not including mutation site residues
            interface_res = self.read_inter_result(interface, partner, mutation_chainid)
            if self.add_mut_to_interface and mutation_crtoken is not None:
                for mut in mutation_crtoken:
                    if mut not in interface_res:
                        interface_res.append(mut)

            if len(interface_res) == 0:  print('no mutation chain is included in the pairwise interface based on provided chains of interest: {}'.format(partner))
            # print(mutation_crtoken, interface_res) # ['V_17'] # ['H_31', 'H_54', 'V_17', 'V_21']
            # besides, both pymol and main_chain_processing in _3_generate_json_input can solve the case of "H_100A", "H_100B", and "H_100C"

            # whether to create a smaller graph from the original complete protein graph with new compact node and edge indices
            # if self.graph_cutoff:
            #     core_crtoken = mutation_crtoken + interface_res
            #     core_crtoken_mask = torch.as_tensor(np.isin(wt_effective_reslist, core_crtoken), device=self.device, dtype=torch.bool)
            # if to further implement this, need to calculate the Euclidean distance between all residue nodes in a protein and the nodes corresponding to core_crtoken_mask
            # for a residue node in a protein, if its distance to any core_crtoken_mask nodes less than the defined graphcutoff, it will be retained and be recorded the position in node feature list (like the relevant function defined in GeoPPI)
            # after getting all residue nodes to be retained, we can use the subgraph function defined below the get the new compact graph for following calculation
            # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html

            wt_effective_reslist = np.array(wt_effective_reslist)
            # mutation_mask = torch.as_tensor(np.where(np.isin(wt_effective_reslist, mutation_crtoken))[0], device=self.device, dtype=torch.int32)
            # interface_mask = torch.as_tensor(np.where(np.isin(wt_effective_reslist, interface_res))[0], device=self.device, dtype=torch.int32)
            # assuming after pre-processing, the cr_token is unique for each residue in a protein (thus np.isin is used)
            # about np.isin(A, B): check whether A is in B
            mutation_mask = torch.as_tensor(np.isin(wt_effective_reslist, mutation_crtoken), device=self.device, dtype=torch.bool)
            interface_mask = torch.as_tensor(np.isin(wt_effective_reslist, interface_res), device=self.device, dtype=torch.bool)
            ddg = torch.as_tensor(ddg, device=self.device, dtype=torch.float32)
            # for PyG, only variable with the name of edge index will automatically increment based on node numbers of each graph, and for features like initial node embedding, only a simple concatenation will be performed
            # print('mutation_mask:', mutation_mask)
            # print('interface_mask:', interface_mask)
            # print('wt_edge_index:', wt_edge_index)
            # print('mt_edge_index:', mt_edge_index)
            # print(mutation_mask.size(), interface_mask.size()) # torch.Size([615]) torch.Size([615])
            assert mutation_mask.size() == interface_mask.size(), 'the sizes of mutation_mask and interface_mask should be the same'

            if self.whether_spatial_graph:
                data = torch_geometric.data.Data(wt_node_s=wt_node_s, wt_node_v=wt_node_v, wt_edge_s=wt_edge_s, wt_edge_v=wt_edge_v, mt_node_s=mt_node_s, mt_node_v=mt_node_v, mt_edge_s=mt_edge_s, mt_edge_v=mt_edge_v,
                                                 wt_edge_index=wt_edge_index, mt_edge_index=mt_edge_index, mutation_mask=mutation_mask, interface_mask=interface_mask, ddg=ddg, mask_size = mutation_mask.size(0),
                                                 wt_extra_edge_s=wt_extra_edge_s, wt_extra_edge_v=wt_extra_edge_v, mt_extra_edge_s=mt_extra_edge_s, mt_extra_edge_v=mt_extra_edge_v,
                                                 wt_extra_edge_index=wt_extra_edge_index, mt_extra_edge_index=mt_extra_edge_index, name=name)
            else:
                data = torch_geometric.data.Data(wt_node_s=wt_node_s, wt_node_v=wt_node_v, wt_edge_s=wt_edge_s, wt_edge_v=wt_edge_v, mt_node_s=mt_node_s, mt_node_v=mt_node_v, mt_edge_s=mt_edge_s, mt_edge_v=mt_edge_v,
                                                 wt_edge_index=wt_edge_index, mt_edge_index=mt_edge_index, mutation_mask=mutation_mask, interface_mask=interface_mask, ddg=ddg, mask_size = mutation_mask.size(0), name=name)

            return data

    # for screening mutation chain related residues in the interface generated by pymol (the mutation chain is included in the screened pairwise interfaces by pymol)
    # if_info (E_I), chainid (chains including mutations)
    def read_inter_result(self, interface, if_info=None, chainid=None, old2new=None):
        # print(if_info, chainid) # HL_VW ['V']
        if if_info is not None:
            info1 = if_info.split('_')
            pA = info1[0]  # chains of interest for protein A
            pB = info1[1]  # chains of interest for protein B

            # construct a mapping between all chains (of interest) in protein A and all chains in protein B
            mappings = {}
            for a in pA:
                for b in pB:
                    if a not in mappings:
                        mappings[a] = [b]
                    else:
                        mappings[a] += [b]
                    if b not in mappings:
                        mappings[b] = [a]
                    else:
                        mappings[b] += [a]

            # print('mappings:', mappings) # mappings: {'H': ['V', 'W'], 'V': ['H', 'L'], 'W': ['H', 'L'], 'L': ['V', 'W']}
            target_chains = []
            for chainidx in chainid: # chain ids for mutation sites, ['V']
                if chainidx in mappings: # whether mutation chain is in the provided chains of interest
                    target_chains += mappings[chainidx]
            # print('target_chains:', target_chains) # target_chains: ['H', 'L']
            # get all corresponding chains (in another protein) of all the mutation chain (in current protein)

            target_inters = []
            for chainidx in chainid:
                target_inters += ['{}_{}'.format(chainidx, y) for y in target_chains] + ['{}_{}'.format(y, chainidx) for y in target_chains] # for the case that y and chainidx are put in different order by pymol
            # print(target_inters) # ['V_H', 'V_L', 'H_V', 'L_V']
            # get all combinations of the mutation chains and corresponding chains (for screening defined interface residues from all interfaces geneerated by pymol (which may not include mutation chains))

            target_inters = list(set(target_inters))
            # print(target_inters) # ['V_L', 'V_H', 'L_V', 'H_V']

        # if partner information is empty
        else:
            target_inters = None

        # open generated interface file (by pymol)
        interlines = interface
        interface_res = []
        for line in interlines: # iterate all pymol interfaces
            iden = line[:3]
            # print(line, iden) # H_L_H_37 H_L

            # only consider the case that target_inters is not empty (interface_res is generated following this rule)
            if target_inters is None:
                if iden.split('_')[0] not in chainid and iden.split('_')[1] not in chainid: # chainid: mutation chain ids
                    continue
                # else: retain pymol interface entries that include mutation chains but are not considered by target_inters
                # I guess that it is for some cases that provided pdb structures not just have chains of interest in original csv mutation file (but these outlier chains still have interfaces with mutation chains)
            else:
                if iden not in target_inters:
                    continue

            infor = line[4:].strip().split('_')  # chainid, resid
            assert len(infor) == 2
            # adding interface position（based on chain id + residue id in this chain）
            interface_res.append('_'.join(infor))

        if old2new is not None:
            mapps = {x[:-4]: y[:-4] for x, y in old2new.items()}
            interface_res = [mapps[x] for x in interface_res if x in mapps]

        return interface_res

    def feature_generator(self, protein):
        name = protein['name']
        seq = protein['seq'] # based on effective/natural AA sequences
        sasa_dict = protein['sasa_dict']
        original_coords = protein['coords']
        original_sidec = copy.deepcopy(protein['sidechain_dict'])
        # residue_list records the cr_token sequentially following the residue order of original pdb file
        residue_list = protein['res_idx_list']
        residue_num = len(residue_list)
        assert residue_num > 0, 'Residue number in current protein should be larger than 0: {}'.format(name)

        sidec_atom_list, atom_set = dict(), set()
        # record side chain atom type (following the residue_list order) and atom weight
        for res in residue_list:
            # this residue (identified by cr_token) has the side chain atoms
            if res in original_sidec.keys():
                # currently atoms in side chain are ordered by 'sorted' function
                atoms = [i[0] for i in np.array(original_sidec[res])[:, 0]]
                # self.atom_mass = {'C': 0.1621, 'N': 0.1891, 'O': 0.2160, 'S': 0.4328} only has these four atoms, if other atoms occur, an error will be raised
                sidec_atom_list[res] = torch.as_tensor([self.atom_mass[j] for j in atoms], device=self.device, dtype=torch.float32).view(-1, 1)
                for atom in atoms:
                    atom_set.add(atom)

        # transform coordinate data into tensor
        for key in original_sidec.keys():
            original_sidec[key] = torch.as_tensor(np.array(original_sidec[key])[:, 1:].astype(np.float32), device=self.device, dtype=torch.float32)
        original_coords = torch.as_tensor(original_coords, device=self.device, dtype=torch.float32)

        # currently, the centroid of current protein is based on the coordinates of all backbone atoms (N, CA, C, O) rather than just based on CA
        original_centroid = self.centroid(original_coords)
        original_coords = original_coords - original_centroid # normalize backbone coordinates
        # for the case of NaN value occurring in atom coordinates (currently try to arise errors)
        mask_nan = torch.isfinite(original_coords.sum(dim=(1, 2)))
        assert (~mask_nan).sum() == 0, 'Current pdb has invalid coordinates.'
        original_X_ca = original_coords[:, 1]
        for key in original_sidec.keys():
            original_sidec[key] = original_sidec[key] - original_centroid # normalize side chain coordinates

        # start to generate residue features based on normalized coordinates
        original_edge_index = torch_cluster.knn_graph(original_X_ca, k=self.top_k)  # knn_graph self loop default: False, the self-loop-like operation is realized in GVPConvLayer (formulas 4-5)
        # edge features (3)
        original_pos_embeddings = self._positional_embeddings(original_edge_index)
        original_E_vectors = original_X_ca[original_edge_index[0]] - original_X_ca[original_edge_index[1]]
        original_rbf = _rbf(original_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
        # if the radius graph is needed
        if self.whether_spatial_graph:
            extra_edge_index = torch_cluster.radius_graph(original_X_ca, r=10.0)
            extra_pos_embeddings = self._positional_embeddings(extra_edge_index)
            extra_E_vectors = original_X_ca[extra_edge_index[0]] - original_X_ca[extra_edge_index[1]]
            extra_rbf = _rbf(extra_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

        # node features
        # the below four type of features are the features specified in original GVP-GNN paper
        original_dihedrals = self._dihedrals(original_coords)
        original_orientations = self._orientations(original_X_ca)  # torch.Size([126, 2, 3])
        original_sidechains = self._sidechains(original_coords)
        # here non-natural residues have been removed in _3_generate_json_input.py (no mutation occurs in non-natural AA sites and all mutations are natural AA mutations)
        # in other words, the non-natural AA sites will be ignored in WT and MT simultaneously
        aatype_onehot = F.one_hot(torch.LongTensor([self.letter_to_num[i] for i in seq if i != ':']), num_classes=20).to(self.device)

        # extra node feature
        # SASA, need to check the numerical range of SASA (currently choose not to normalize SASA)
        SASA = torch.as_tensor([float(sasa_dict[i]['asa_complex']) for i in residue_list], device=self.device, dtype=torch.float32)  # every residue will be equipped with a SASA and an interface value
        # whether in complex interface or not
        interface = torch.as_tensor([int(sasa_dict[i]['interface']) for i in residue_list], device=self.device, dtype=torch.int32)
        # print(len(interface), len(residue_list), residue_list) # 619 619 ['H_1', 'H_2', 'H_3', 'H_4', 'H_5'
        # *** elements in interface should correspond to the order in residue_list (protein['res_idx_list']) ***
        # *** the use of interface information can be obtained from independently provided SASA files or pymol (the results should be basically the same) ***

        # which chain current residue locates to
        chainid_seq = []
        for i in range(len(seq.split(':'))): # seq: based on effective/natural AA sequences
            chainid_seq.extend([i] * len(seq.split(':')[i]))
        chainid_seq = torch.as_tensor(chainid_seq, device=self.device, dtype=torch.int32) / i  # dividing i is for normalization
        assert SASA.size() == interface.size() == chainid_seq.size(), 'SASA, interface, chainid_seq should have the same size in {}.'.format(name)

        # residue_list records the cr_token sequentially following the residue order of original pdb file
        # currently the below four four chain coordinate information is not normalized and the vacant places are filled with zero
        if self.sidec_chain_normalization:
            # current centroid of protein backbone complex is calculated based on original_coords (C+CA+N+O), thus the further normalization is also based on this
            # currently original_coords has been through centroid normalization
            length = torch.sqrt(torch.sum((original_coords.view(-1, 3) ** 2), -1))
            length = length[torch.argmax(length)]
            sidec_seq_max = (torch.concat([torch.max(original_sidec[i], dim=0, keepdim=True)[0] if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length)\
                .type(torch.float32).to(self.device)
            sidec_seq_centroid = (torch.concat([torch.mean(original_sidec[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length)\
                .type(torch.float32).to(self.device)
            sidec_seq_mass = (torch.concat([torch.mean(original_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length)\
                .type(torch.float32).to(self.device)
            sidec_CA_relations = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list) # relative value, not influenced by the further normlization
            original_X_ca = original_X_ca / length
        else:
            sidec_seq_max = torch.concat(
                [torch.max(original_sidec[i], dim=0, keepdim=True)[0] if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]
            ).type(torch.float32).to(self.device)
            sidec_seq_centroid = torch.concat(
                [torch.mean(original_sidec[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]
            ).type(torch.float32).to(self.device)
            sidec_seq_mass = torch.concat( # the below i is the cr_token key in res_idx_list
                [torch.mean(original_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list] # also based on original_centroid
            ).type(torch.float32).to(self.device)
            sidec_CA_relations = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list)
        assert aatype_onehot.size(0) == SASA.size(0) == original_dihedrals.size(0) == original_orientations.size(0) == original_sidechains.size(0) == interface.size(0) \
               == sidec_seq_max.size(0) == sidec_seq_centroid.size(0) == sidec_seq_mass.size(0) == sidec_CA_relations.size(0) == chainid_seq.size(0) == original_X_ca.size(0), 'All generated features does not have the same length.'

        # extra CG features
        if self.whether_CG_feature:
            # cg_feature is based on res_idx_list, i.e., residue_list here, same to the retrieve of the backbone coordinates
            cg_feature = protein['cg_feature']
            pos_feats_num = torch.as_tensor([9, 12, 6, 3, 2], device=self.device, dtype=torch.int64)  # feature number for each position of cg features (5 pos in total)
            if self.CG_encoding_type == 'sincos':
                # normalize each column/feature separately using pos_feats_num
                cg_feature = (torch.as_tensor(cg_feature, device=self.device, dtype=torch.float32) / pos_feats_num) * 6.283
                cg_feature = torch.cat([torch.sin(cg_feature), torch.cos(cg_feature)], dim=-1)
            elif self.CG_encoding_type == 'onehot':
                # old version:
                # cg_feature = F.one_hot(torch.LongTensor(cg_feature), num_classes=18).to(self.device)
                # cg_feature = cg_feature.view(cg_feature.size(0), -1)
                # new version:
                cg_feature = torch.LongTensor(cg_feature).to(self.device) # F.one_hot only takes int64 as the input
                cg_feature = torch.cat([F.one_hot(cg_feature[:, i], num_classes=pos_feats_num[i]) for i in range(pos_feats_num.size(0))], dim=-1).float()
            else:
                cg_feature = torch.as_tensor(cg_feature, device=self.device, dtype=torch.float32) / pos_feats_num

            assert aatype_onehot.size(0) == cg_feature.size(0)
            node_s = torch.cat([aatype_onehot, SASA.unsqueeze(-1), original_dihedrals, interface.unsqueeze(-1), chainid_seq.unsqueeze(-1), cg_feature], dim=-1)
        else:
            node_s = torch.cat([aatype_onehot, SASA.unsqueeze(-1), original_dihedrals, interface.unsqueeze(-1), chainid_seq.unsqueeze(-1)], dim=-1)

        node_v = torch.cat([original_orientations, original_sidechains.unsqueeze(-2), sidec_seq_max.unsqueeze(-2), sidec_seq_centroid.unsqueeze(-2), sidec_seq_mass.unsqueeze(-2), sidec_CA_relations, original_X_ca.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([original_rbf, original_pos_embeddings], dim=-1)
        edge_v = _normalize(original_E_vectors).unsqueeze(-2)

        if self.whether_spatial_graph:
            extra_edge_s = torch.cat([extra_rbf, extra_pos_embeddings], dim=-1)
            extra_edge_v = _normalize(extra_E_vectors).unsqueeze(-2)
            node_s, node_v, edge_s, edge_v, extra_edge_s, extra_edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v, extra_edge_s, extra_edge_v))
            return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list, extra_edge_index, extra_edge_s, extra_edge_v
        else:
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
            return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list

    def centroid(self, coords):
        """
        Centroid is the mean position of all the points in all of the coordinate
        directions, from a vectorset X.
        https://en.wikipedia.org/wiki/Centroid
        C = sum(X)/len(X)
        Parameters
        ----------
        X : array
            (N,D) matrix, where N is points and D is dimension.
        Returns
        """
        coords = coords.view(-1, 3).mean(axis=0)
        return coords

    def _positional_embeddings(self, edge_index, num_embeddings=None, period_range=[2, 1000]):

        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings  # 16
        d = edge_index[0] - edge_index[1]
        # print(d.size()) torch.Size([11880])

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        # print(frequency.size()) torch.Size([8])
        angles = d.unsqueeze(-1) * frequency
        # print(angles.size()) torch.Size([11880, 8])
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        # print(E.size()) torch.Size([11880, 16])
        return E

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        # X: torch.Size([166, 4, 3]) # backbone atom coordinates for current protein
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        # print(X.size()) torch.Size([534, 3]) # remove O coordinates, and flatten the rest of atom coordinates (N，Ca，C)
        dX = X[1:] - X[:-1] # Ca1, C1, N2, Ca2, C2, N3, Ca3, C3 # N1, Ca1, C1, N2, Ca2, C2, N3, Ca3

        # print(X.size(), dX.size()) torch.Size([330, 3]) torch.Size([329, 3])
        U = _normalize(dX, dim=-1)

        u_2 = U[:-2] # these operations are used in the sample dimension
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1) # calculate the cross product, and then do the normalization for it
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        # sign函数（-1-0-1）（感觉用处主要是为后面返回的弧度区分正负（根据在化学中使用二面角的定义），https://en.wikipedia.org/wiki/Dihedral_angle），acos：computes the inverse cosine of each element in input（返回的是弧度）
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        # print(D.size()) torch.Size([552])

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) # 相当于在生成的一维D（各二面角）的基础上在最前面填一个0，在最后填两个0，感觉就是在整个计算过程中，对于一个蛋白质，计算其中各氨基酸的二面角时，缺少了phi[0], psi[-1], omega[-1]，然后使用0进行填补
        # print(D.size(), D) torch.Size([555])

        D = torch.reshape(D, [-1, 3]) # 将φi, ψi, ωi的特征赋给每一个氨基酸（在当前scheme下，所有蛋白质的第一个氨基酸和最后的一个氨基酸会一共缺失三个特征：phi[0], psi[-1], omega[-1]）
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        # print(D_features.size(), X.size()) # torch.Size([330, 6]) torch.Size([990, 3]) # 应该是为每一个蛋白质的氨基酸分配一个维度为6的特征
        return D_features

    def _orientations(self, X):
        # print(X[1:].size(), X[:-1].size()) # torch.Size([125, 3]) torch.Size([125, 3])
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        # print(forward.size())
        # print(backward.size())
        forward = F.pad(forward, [0, 0, 0, 1])  # [274, 3]，在最下面填一层0，因为感觉第一个值没有计算
        backward = F.pad(backward, [0, 0, 1, 0])  # [274, 3]，在最上面填一层0，最后一个值没有计算
        # print(forward.size())  # [275, 3]
        # print(backward.size())  # [275, 3]
        # print(X.size()) # [275, 3]
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        # torch.Size([166, 4, 3]) (for N, Ca, C, O)
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    # currently _CA_to_sidechains is based on normalized CA coordinates instead of the absolute ones
    # sidec_CA_relations = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list)
    def _CA_to_sidechains(self, CA, sidec, residue_list):
        assert CA.size(0) == len(residue_list) # one CA corresponds to one residue_list element
        vec = []
        for i in range(len(residue_list)):
            if residue_list[i] in sidec.keys():
                forward = torch.mean(_normalize(sidec[residue_list[i]] - CA[i]), dim=0, keepdim=True)
                backward = torch.mean(_normalize(CA[i] - sidec[residue_list[i]]), dim=0, keepdim=True)
                overall = torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
                vec.append(overall)
            else:
                overall = torch.zeros(1, 2, 3).to(self.device)
                vec.append(overall)

        return torch.cat(vec, dim=0)


# rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    # print('RBF.size():', RBF.size()) # torch.Size([2460, 16])
    return RBF


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    # Replaces NaN, positive infinity, and negative infinity values in input with the values specified by nan, posinf, and neginf, respectively.
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


# after defining source data class, we need to define relevant pytorch Dataset class to produce required node and edge features
# here we do not need to add noises to protein graphs, but we need to consider how to generate features for widetype and mutation protein graphs at the same time
class FinetuningGraphDataset_wosidec(data.Dataset):
    def __init__(self, data_list, num_positional_embeddings=16, top_k = 30, num_rbf=16, device='cpu', whether_spatial_graph = False, add_mut_to_interface=False, remove_SASA_all=False):
        super(FinetuningGraphDataset_wosidec, self).__init__()

        self.data_list = data_list # training/val list that stores each entry in the format of dict
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(''.join(e['widetype_complex']['seq'].split(':'))) for e in data_list] # used for passing residue node number to PyG DataLoader
        self.add_mut_to_interface = add_mut_to_interface
        self.remove_SASA_all = remove_SASA_all

        # we can limit the residue number of a protein in pytorch DataLoader (by detecting and removing it through the set threshold)

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.atom_mass = {'C': 0.1621, 'N': 0.1891, 'O': 0.2160, 'S': 0.4328}
        # temp = 12.011 + 14.0067 + 15.9994 + 32.06
        # print(12.011/temp, 14.0067/temp, 15.9994/temp, 32.06/temp)

        self.whether_spatial_graph = whether_spatial_graph

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        # The __getitem__ function loads and returns a sample from the dataset at the given index idx.
        return self._featurize_as_graph(self.data_list[i], i)

    # *** need to check whether the generated mask and random coordinate perturbation will change in different epochs ***
    # *** need to ensure no operation will influence the original data stored in self.data_list (protein in _featurize_as_graph in originated from self.data_list[i]) ***
    # *** otherwise the data generated for the next batch will be changed, causing errors ***
    def _featurize_as_graph(self, protein, i):
        # name, widetype_complex, mutation_complex, mutation_info, interface, ddg
        # widetype_complex, mutation_complex include: (seq, coords, num_chains, name, sidechain_dict, sasa_dict, res_idx_list, len_complete_aalist)
        name = protein['name'] # name: widetype protein name + serial number in original mutation file
        with torch.no_grad():
            # here we can mark the interface and mutation residues (and can set a range threshold for them to record an area around interface and mutation sites)
            partner = protein['partner']
            wt = protein['widetype_complex']
            mt = protein['mutation_complex']

            wt_features, mt_features = map(self.feature_generator, (wt, mt))

            if self.whether_spatial_graph:
                # return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list, extra_edge_index, extra_edge_s, extra_edge_v
                wt_node_s, wt_node_v, wt_edge_s, wt_edge_v, wt_edge_index, wt_effective_reslist, wt_extra_edge_index, wt_extra_edge_s, wt_extra_edge_v = \
                    wt_features[0], wt_features[1], wt_features[2], wt_features[3], wt_features[4], wt_features[5], wt_features[6], wt_features[7], wt_features[8]
                mt_node_s, mt_node_v, mt_edge_s, mt_edge_v, mt_edge_index, mt_effective_reslist, mt_extra_edge_index, mt_extra_edge_s, mt_extra_edge_v = \
                    mt_features[0], mt_features[1], mt_features[2], mt_features[3], mt_features[4], mt_features[5], mt_features[6], mt_features[7], mt_features[8]
            else:
                wt_node_s, wt_node_v, wt_edge_s, wt_edge_v, wt_edge_index, wt_effective_reslist = wt_features[0], wt_features[1], wt_features[2], wt_features[3], wt_features[4], wt_features[5]
                mt_node_s, mt_node_v, mt_edge_s, mt_edge_v, mt_edge_index, mt_effective_reslist = mt_features[0], mt_features[1], mt_features[2], mt_features[3], mt_features[4], mt_features[5]

            assert wt_node_s.size() == mt_node_s.size() and wt_node_v.size() == mt_node_v.size() and len(wt_effective_reslist) == len(mt_effective_reslist) and\
                   wt_node_s.size(0) == len(wt_effective_reslist), 'widetype and mutation complexes have different residue node feature number and dim'

            # for GeoPPI, the objective for it to calculate mean and max pooling is the mutation sites and interface sites (including mutation sites)
            # thus, what we can provide here are masks for mutation sites and interface sites, and corresponding mean and max pooling operations are put into finetuning models
            mutation_info, interface, ddg = protein['mutation_info'], protein['interface'], protein['ddg']
            # mutation_info: V:F17A, interface: ['H_L_H_37', 'H_L_H_39'], ddg: 0.0
            # print(wt_effective_reslist) # 'H_221', 'H_222', 'H_223', 'H_224', 'L_1', 'L_2', 'L_3', 'L_4'

            # generate mutation and interface site masks
            mutation_crtoken = ['{}_{}'.format(i.split(':')[0], i.split(':')[-1][1:-1]) for i in mutation_info.split(',')]
            mutation_chainid = [i.split(':')[0] for i in mutation_info.split(',')]
            # for screening mutation chain related residues in the interface generated by pymol (the mutation chain is included in the screened pairwise interfaces by pymol)
            # input: 1. pymol generated interface, 2. chain of interests provided by mutation file, 3. mutation chain
            # output: not including mutation site residues
            interface_res = self.read_inter_result(interface, partner, mutation_chainid)
            if self.add_mut_to_interface and mutation_crtoken is not None:
                for mut in mutation_crtoken:
                    if mut not in interface_res:
                        interface_res.append(mut)

            if len(interface_res) == 0:  print('no mutation chain is included in the pairwise interface based on provided chains of interest: {}'.format(partner))
            # print(mutation_crtoken, interface_res) # ['V_17'] # ['H_31', 'H_54', 'V_17', 'V_21']
            # besides, both pymol and main_chain_processing in _3_generate_json_input can solve the case of "H_100A", "H_100B", and "H_100C"

            # whether to create a smaller graph from the original complete protein graph with new compact node and edge indices
            # if self.graph_cutoff:
            #     core_crtoken = mutation_crtoken + interface_res
            #     core_crtoken_mask = torch.as_tensor(np.isin(wt_effective_reslist, core_crtoken), device=self.device, dtype=torch.bool)
            # if to further implement this, need to calculate the Euclidean distance between all residue nodes in a protein and the nodes corresponding to core_crtoken_mask
            # for a residue node in a protein, if its distance to any core_crtoken_mask nodes less than the defined graphcutoff, it will be retained and be recorded the position in node feature list (like the relevant function defined in GeoPPI)
            # after getting all residue nodes to be retained, we can use the subgraph function defined below the get the new compact graph for following calculation
            # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html

            wt_effective_reslist = np.array(wt_effective_reslist)
            # mutation_mask = torch.as_tensor(np.where(np.isin(wt_effective_reslist, mutation_crtoken))[0], device=self.device, dtype=torch.int32)
            # interface_mask = torch.as_tensor(np.where(np.isin(wt_effective_reslist, interface_res))[0], device=self.device, dtype=torch.int32)
            # assuming after pre-processing, the cr_token is unique for each residue in a protein (thus np.isin is used)
            # about np.isin(A, B): check whether A is in B
            mutation_mask = torch.as_tensor(np.isin(wt_effective_reslist, mutation_crtoken), device=self.device, dtype=torch.bool)
            interface_mask = torch.as_tensor(np.isin(wt_effective_reslist, interface_res), device=self.device, dtype=torch.bool)
            ddg = torch.as_tensor(ddg, device=self.device, dtype=torch.float32)
            # for PyG, only variable with the name of edge index will automatically increment based on node numbers of each graph, and for features like initial node embedding, only a simple concatenation will be performed
            # print('mutation_mask:', mutation_mask)
            # print('interface_mask:', interface_mask)
            # print('wt_edge_index:', wt_edge_index)
            # print('mt_edge_index:', mt_edge_index)
            # print(mutation_mask.size(), interface_mask.size()) # torch.Size([615]) torch.Size([615])
            assert mutation_mask.size() == interface_mask.size(), 'the sizes of mutation_mask and interface_mask should be the same'

            if self.whether_spatial_graph:
                data = torch_geometric.data.Data(wt_node_s=wt_node_s, wt_node_v=wt_node_v, wt_edge_s=wt_edge_s, wt_edge_v=wt_edge_v, mt_node_s=mt_node_s, mt_node_v=mt_node_v, mt_edge_s=mt_edge_s, mt_edge_v=mt_edge_v,
                                                 wt_edge_index=wt_edge_index, mt_edge_index=mt_edge_index, mutation_mask=mutation_mask, interface_mask=interface_mask, ddg=ddg, mask_size = mutation_mask.size(0),
                                                 wt_extra_edge_s=wt_extra_edge_s, wt_extra_edge_v=wt_extra_edge_v, mt_extra_edge_s=mt_extra_edge_s, mt_extra_edge_v=mt_extra_edge_v,
                                                 wt_extra_edge_index=wt_extra_edge_index, mt_extra_edge_index=mt_extra_edge_index, name=name)
            else:
                data = torch_geometric.data.Data(wt_node_s=wt_node_s, wt_node_v=wt_node_v, wt_edge_s=wt_edge_s, wt_edge_v=wt_edge_v, mt_node_s=mt_node_s, mt_node_v=mt_node_v, mt_edge_s=mt_edge_s, mt_edge_v=mt_edge_v,
                                                 wt_edge_index=wt_edge_index, mt_edge_index=mt_edge_index, mutation_mask=mutation_mask, interface_mask=interface_mask, ddg=ddg, mask_size = mutation_mask.size(0), name=name)

            return data

    # for screening mutation chain related residues in the interface generated by pymol (the mutation chain is included in the screened pairwise interfaces by pymol)
    # if_info (E_I), chainid (chains including mutations)
    # # input: 1. pymol generated interface, 2. chain of interests provided by mutation file, 3. mutation chain
    def read_inter_result(self, interface, if_info=None, chainid=None, old2new=None):
        # print(if_info, chainid) # HL_VW ['V']
        if if_info is not None:
            info1 = if_info.split('_')
            pA = info1[0]  # chains of interest for protein A
            pB = info1[1]  # chains of interest for protein B

            # construct a mapping between all chains (of interest) in protein A and all chains in protein B
            mappings = {}
            for a in pA:
                for b in pB:
                    if a not in mappings:
                        mappings[a] = [b]
                    else:
                        mappings[a] += [b]
                    if b not in mappings:
                        mappings[b] = [a]
                    else:
                        mappings[b] += [a]

            # print('mappings:', mappings) # mappings: {'H': ['V', 'W'], 'V': ['H', 'L'], 'W': ['H', 'L'], 'L': ['V', 'W']}
            target_chains = []
            for chainidx in chainid: # chain ids for mutation sites, ['V']
                if chainidx in mappings: # whether mutation chain is in the provided chains of interest
                    target_chains += mappings[chainidx]
            # print('target_chains:', target_chains) # target_chains: ['H', 'L']
            # get all corresponding chains (in another protein) of all the mutation chain (in current protein)

            target_inters = []
            for chainidx in chainid:
                target_inters += ['{}_{}'.format(chainidx, y) for y in target_chains] + ['{}_{}'.format(y, chainidx) for y in target_chains] # for the case that y and chainidx are put in different order by pymol
            # print(target_inters) # ['V_H', 'V_L', 'H_V', 'L_V']
            # get all combinations of the mutation chains and corresponding chains (for screening defined interface residues from all interfaces geneerated by pymol (which may not include mutation chains))

            target_inters = list(set(target_inters))
            # print(target_inters) # ['V_L', 'V_H', 'L_V', 'H_V']

        # if partner information is empty
        else:
            target_inters = None

        # open generated interface file (by pymol)
        interlines = interface
        interface_res = []
        for line in interlines: # iterate all pymol interfaces
            iden = line[:3]
            # print(line, iden) # H_L_H_37 H_L

            # only consider the case that target_inters is not empty (interface_res is generated following this rule)
            if target_inters is None:
                if iden.split('_')[0] not in chainid and iden.split('_')[1] not in chainid: # chainid: mutation chain ids
                    continue
                # else: retain pymol interface entries that include mutation chains but are not considered by target_inters
                # I guess that it is for some cases that provided pdb structures not just have chains of interest in original csv mutation file (but these outlier chains still have interfaces with mutation chains)
            else:
                if iden not in target_inters:
                    continue

            infor = line[4:].strip().split('_')  # chainid, resid
            assert len(infor) == 2
            # adding interface position（based on chain id + residue id in this chain）
            interface_res.append('_'.join(infor))

        if old2new is not None:
            mapps = {x[:-4]: y[:-4] for x, y in old2new.items()}
            interface_res = [mapps[x] for x in interface_res if x in mapps]

        return interface_res

    def feature_generator(self, protein):
        name = protein['name']
        seq = protein['seq'] # based on effective/natural AA sequences
        sasa_dict = protein['sasa_dict']
        original_coords = protein['coords']
        # residue_list records the cr_token sequentially following the residue order of original pdb file
        residue_list = protein['res_idx_list']
        residue_num = len(residue_list)
        assert residue_num > 0, 'Residue number in current protein should be larger than 0: {}'.format(name)

        # transform coordinate data into tensor
        original_coords = torch.as_tensor(original_coords, device=self.device, dtype=torch.float32)

        # currently, the centroid of current protein is based on the coordinates of all backbone atoms (N, CA, C, O) rather than just based on CA
        original_centroid = self.centroid(original_coords)
        original_coords = original_coords - original_centroid # normalize backbone coordinates
        # for the case of NaN value occurring in atom coordinates (currently try to arise errors)
        mask_nan = torch.isfinite(original_coords.sum(dim=(1, 2)))
        assert (~mask_nan).sum() == 0, 'Current pdb has invalid coordinates.'
        original_X_ca = original_coords[:, 1]

        # start to generate residue features based on normalized coordinates
        original_edge_index = torch_cluster.knn_graph(original_X_ca, k=self.top_k)  # knn_graph self loop default: False, the self-loop-like operation is realized in GVPConvLayer (formulas 4-5)
        # edge features (3)
        original_pos_embeddings = self._positional_embeddings(original_edge_index)
        original_E_vectors = original_X_ca[original_edge_index[0]] - original_X_ca[original_edge_index[1]]
        original_rbf = _rbf(original_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
        # if the radius graph is needed
        if self.whether_spatial_graph:
            extra_edge_index = torch_cluster.radius_graph(original_X_ca, r=10.0)
            extra_pos_embeddings = self._positional_embeddings(extra_edge_index)
            extra_E_vectors = original_X_ca[extra_edge_index[0]] - original_X_ca[extra_edge_index[1]]
            extra_rbf = _rbf(extra_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

        # node features
        # the below four type of features are the features specified in original GVP-GNN paper
        original_dihedrals = self._dihedrals(original_coords)
        original_orientations = self._orientations(original_X_ca)  # torch.Size([126, 2, 3])
        original_sidechains = self._sidechains(original_coords)
        # here non-natural residues have been removed in _3_generate_json_input.py (no mutation occurs in non-natural AA sites and all mutations are natural AA mutations)
        # in other words, the non-natural AA sites will be ignored in WT and MT simultaneously
        aatype_onehot = F.one_hot(torch.LongTensor([self.letter_to_num[i] for i in seq if i != ':']), num_classes=20).to(self.device)

        # extra node feature
        # SASA, need to check the numerical range of SASA (currently choose not to normalize SASA)
        SASA = torch.as_tensor([float(sasa_dict[i]['asa_complex']) for i in residue_list], device=self.device, dtype=torch.float32)  # every residue will be equipped with a SASA and an interface value
        # whether in complex interface or not
        interface = torch.as_tensor([int(sasa_dict[i]['interface']) for i in residue_list], device=self.device, dtype=torch.int32)
        # print(len(interface), len(residue_list), residue_list) # 619 619 ['H_1', 'H_2', 'H_3', 'H_4', 'H_5'
        # *** elements in interface should correspond to the order in residue_list (protein['res_idx_list']) ***
        # *** the use of interface information can be obtained from independently provided SASA files or pymol (the results should be basically the same) ***

        # which chain current residue locates to
        chainid_seq = []
        for i in range(len(seq.split(':'))): # seq: based on effective/natural AA sequences
            chainid_seq.extend([i] * len(seq.split(':')[i]))
        chainid_seq = torch.as_tensor(chainid_seq, device=self.device, dtype=torch.int32) / i  # dividing i is for normalization
        assert SASA.size() == interface.size() == chainid_seq.size(), 'SASA, interface, chainid_seq should have the same size in {}.'.format(name)

        assert aatype_onehot.size(0) == SASA.size(0) == original_dihedrals.size(0) == original_orientations.size(0) == original_sidechains.size(0) == interface.size(0) \
               == chainid_seq.size(0) == original_X_ca.size(0), 'All generated features does not have the same length.'

        # merge all information and return it
        if self.remove_SASA_all == True:
            node_s = torch.cat([aatype_onehot, original_dihedrals, chainid_seq.unsqueeze(-1)], dim=-1)
        else:
            node_s = torch.cat([aatype_onehot, SASA.unsqueeze(-1), original_dihedrals, interface.unsqueeze(-1), chainid_seq.unsqueeze(-1)], dim=-1)
        node_v = torch.cat([original_orientations, original_sidechains.unsqueeze(-2), original_X_ca.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([original_rbf, original_pos_embeddings], dim=-1)
        edge_v = _normalize(original_E_vectors).unsqueeze(-2)

        if self.whether_spatial_graph:
            extra_edge_s = torch.cat([extra_rbf, extra_pos_embeddings], dim=-1)
            extra_edge_v = _normalize(extra_E_vectors).unsqueeze(-2)
            node_s, node_v, edge_s, edge_v, extra_edge_s, extra_edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v, extra_edge_s, extra_edge_v))
            return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list, extra_edge_index, extra_edge_s, extra_edge_v
        else:
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
            return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list

    def centroid(self, coords):
        """
        Centroid is the mean position of all the points in all of the coordinate
        directions, from a vectorset X.
        https://en.wikipedia.org/wiki/Centroid
        C = sum(X)/len(X)
        Parameters
        ----------
        X : array
            (N,D) matrix, where N is points and D is dimension.
        Returns
        """
        coords = coords.view(-1, 3).mean(axis=0)
        return coords

    def _positional_embeddings(self, edge_index, num_embeddings=None, period_range=[2, 1000]):

        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings  # 16
        d = edge_index[0] - edge_index[1]
        # print(d.size()) torch.Size([11880])

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        # print(frequency.size()) torch.Size([8])
        angles = d.unsqueeze(-1) * frequency
        # print(angles.size()) torch.Size([11880, 8])
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        # print(E.size()) torch.Size([11880, 16])
        return E

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        # X: torch.Size([166, 4, 3]) # backbone atom coordinates for current protein
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        # print(X.size()) torch.Size([534, 3]) # remove O coordinates, and flatten the rest of atom coordinates (N，Ca，C)
        dX = X[1:] - X[:-1] # Ca1, C1, N2, Ca2, C2, N3, Ca3, C3 # N1, Ca1, C1, N2, Ca2, C2, N3, Ca3

        # print(X.size(), dX.size()) torch.Size([330, 3]) torch.Size([329, 3])
        U = _normalize(dX, dim=-1)

        u_2 = U[:-2] # these operations are used in the sample dimension
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1) # calculate the cross product, and then do the normalization for it
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        # sign函数（-1-0-1）（感觉用处主要是为后面返回的弧度区分正负（根据在化学中使用二面角的定义），https://en.wikipedia.org/wiki/Dihedral_angle），acos：computes the inverse cosine of each element in input（返回的是弧度）
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        # print(D.size()) torch.Size([552])

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) # 相当于在生成的一维D（各二面角）的基础上在最前面填一个0，在最后填两个0，感觉就是在整个计算过程中，对于一个蛋白质，计算其中各氨基酸的二面角时，缺少了phi[0], psi[-1], omega[-1]，然后使用0进行填补
        # print(D.size(), D) torch.Size([555])

        D = torch.reshape(D, [-1, 3]) # 将φi, ψi, ωi的特征赋给每一个氨基酸（在当前scheme下，所有蛋白质的第一个氨基酸和最后的一个氨基酸会一共缺失三个特征：phi[0], psi[-1], omega[-1]）
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        # print(D_features.size(), X.size()) # torch.Size([330, 6]) torch.Size([990, 3]) # 应该是为每一个蛋白质的氨基酸分配一个维度为6的特征
        return D_features

    def _orientations(self, X):
        # print(X[1:].size(), X[:-1].size()) # torch.Size([125, 3]) torch.Size([125, 3])
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        # print(forward.size())
        # print(backward.size())
        forward = F.pad(forward, [0, 0, 0, 1])  # [274, 3]，在最下面填一层0，因为感觉第一个值没有计算
        backward = F.pad(backward, [0, 0, 1, 0])  # [274, 3]，在最上面填一层0，最后一个值没有计算
        # print(forward.size())  # [275, 3]
        # print(backward.size())  # [275, 3]
        # print(X.size()) # [275, 3]
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        # torch.Size([166, 4, 3]) (for N, Ca, C, O)
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    # currently _CA_to_sidechains is based on normalized CA coordinates instead of the absolute ones
    def _CA_to_sidechains(self, CA, sidec, residue_list):
        assert CA.size(0) == len(residue_list) # one CA corresponds to one residue_list element
        vec = []
        for i in range(len(residue_list)):
            if residue_list[i] in sidec.keys():
                forward = torch.mean(_normalize(sidec[residue_list[i]] - CA[i]), dim=0, keepdim=True)
                backward = torch.mean(_normalize(CA[i] - sidec[residue_list[i]]), dim=0, keepdim=True)
                overall = torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
                vec.append(overall)
            else:
                overall = torch.zeros(1, 2, 3).to(self.device)
                vec.append(overall)

        return torch.cat(vec, dim=0)


class FinetuningGraphDataset_wosasa(data.Dataset):
    def __init__(self, data_list, num_positional_embeddings=16, top_k = 30, num_rbf=16, device='cpu', sidec_chain_normalization = False,
                 whether_spatial_graph = False, add_mut_to_interface = False, whether_CG_feature=False, CG_encoding_type='sincos'):
        super(FinetuningGraphDataset_wosasa, self).__init__()

        self.data_list = data_list # training/val list that stores each entry in the format of dict
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(''.join(e['widetype_complex']['seq'].split(':'))) for e in data_list] # used for passing residue node number to PyG DataLoader
        self.add_mut_to_interface = add_mut_to_interface # indicate whether to add mutation cr-token to interface cr_token set

        # we can limit the residue number of a protein in pytorch DataLoader (by detecting and removing it through the set threshold)

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.atom_mass = {'C': 0.1621, 'N': 0.1891, 'O': 0.2160, 'S': 0.4328}
        # temp = 12.011 + 14.0067 + 15.9994 + 32.06
        # print(12.011/temp, 14.0067/temp, 15.9994/temp, 32.06/temp)

        self.sidec_chain_normalization = sidec_chain_normalization # for further normalization of side chain coordinates
        self.whether_spatial_graph = whether_spatial_graph
        self.whether_CG_feature = whether_CG_feature
        self.CG_encoding_type = CG_encoding_type # 'sin'/'onehot'/None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        # The __getitem__ function loads and returns a sample from the dataset at the given index idx.
        return self._featurize_as_graph(self.data_list[i], i)

    # *** need to check whether the generated mask and random coordinate perturbation will change in different epochs ***
    # *** need to ensure no operation will influence the original data stored in self.data_list (protein in _featurize_as_graph in originated from self.data_list[i]) ***
    # *** otherwise the data generated for the next batch will be changed, causing errors ***
    def _featurize_as_graph(self, protein, i):
        # name, widetype_complex, mutation_complex, mutation_info, interface, ddg
        # widetype_complex, mutation_complex include: (seq, coords, num_chains, name, sidechain_dict, sasa_dict, res_idx_list, len_complete_aalist)
        name = protein['name'] # name: widetype protein name + serial number in original mutation file
        with torch.no_grad():
            # here we can mark the interface and mutation residues (and can set a range threshold for them to record an area around interface and mutation sites)
            partner = protein['partner']
            wt = protein['widetype_complex']
            mt = protein['mutation_complex']

            wt_features, mt_features = map(self.feature_generator, (wt, mt))

            if self.whether_spatial_graph:
                # return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list, extra_edge_index, extra_edge_s, extra_edge_v
                wt_node_s, wt_node_v, wt_edge_s, wt_edge_v, wt_edge_index, wt_effective_reslist, wt_extra_edge_index, wt_extra_edge_s, wt_extra_edge_v = \
                    wt_features[0], wt_features[1], wt_features[2], wt_features[3], wt_features[4], wt_features[5], wt_features[6], wt_features[7], wt_features[8]
                mt_node_s, mt_node_v, mt_edge_s, mt_edge_v, mt_edge_index, mt_effective_reslist, mt_extra_edge_index, mt_extra_edge_s, mt_extra_edge_v = \
                    mt_features[0], mt_features[1], mt_features[2], mt_features[3], mt_features[4], mt_features[5], mt_features[6], mt_features[7], mt_features[8]
            else:
                wt_node_s, wt_node_v, wt_edge_s, wt_edge_v, wt_edge_index, wt_effective_reslist = wt_features[0], wt_features[1], wt_features[2], wt_features[3], wt_features[4], wt_features[5]
                mt_node_s, mt_node_v, mt_edge_s, mt_edge_v, mt_edge_index, mt_effective_reslist = mt_features[0], mt_features[1], mt_features[2], mt_features[3], mt_features[4], mt_features[5]

            assert wt_node_s.size() == mt_node_s.size() and wt_node_v.size() == mt_node_v.size() and len(wt_effective_reslist) == len(mt_effective_reslist) and\
                   wt_node_s.size(0) == len(wt_effective_reslist), 'widetype and mutation complexes have different residue node feature number and dim'

            # for GeoPPI, the objective for it to calculate mean and max pooling is the mutation sites and interface sites (including mutation sites)
            # thus, what we can provide here are masks for mutation sites and interface sites, and corresponding mean and max pooling operations are put into finetuning models
            mutation_info, interface, ddg = protein['mutation_info'], protein['interface'], protein['ddg']
            # mutation_info: V:F17A, interface: ['H_L_H_37', 'H_L_H_39'], ddg: 0.0
            # print(wt_effective_reslist) # 'H_221', 'H_222', 'H_223', 'H_224', 'L_1', 'L_2', 'L_3', 'L_4'

            # generate mutation and interface site masks
            mutation_crtoken = ['{}_{}'.format(i.split(':')[0], i.split(':')[-1][1:-1]) for i in mutation_info.split(',')]
            mutation_chainid = [i.split(':')[0] for i in mutation_info.split(',')]
            # for screening mutation chain related residues in the interface generated by pymol (the mutation chain is included in the screened pairwise interfaces by pymol)
            # input: 1. pymol generated interface, 2. chain of interests provided by mutation file, 3. mutation chain
            # output: not including mutation site residues
            interface_res = self.read_inter_result(interface, partner, mutation_chainid)
            if self.add_mut_to_interface and mutation_crtoken is not None:
                for mut in mutation_crtoken:
                    if mut not in interface_res:
                        interface_res.append(mut)

            if len(interface_res) == 0:  print('no mutation chain is included in the pairwise interface based on provided chains of interest: {}'.format(partner))
            # print(mutation_crtoken, interface_res) # ['V_17'] # ['H_31', 'H_54', 'V_17', 'V_21']
            # besides, both pymol and main_chain_processing in _3_generate_json_input can solve the case of "H_100A", "H_100B", and "H_100C"

            # whether to create a smaller graph from the original complete protein graph with new compact node and edge indices
            # if self.graph_cutoff:
            #     core_crtoken = mutation_crtoken + interface_res
            #     core_crtoken_mask = torch.as_tensor(np.isin(wt_effective_reslist, core_crtoken), device=self.device, dtype=torch.bool)
            # if to further implement this, need to calculate the Euclidean distance between all residue nodes in a protein and the nodes corresponding to core_crtoken_mask
            # for a residue node in a protein, if its distance to any core_crtoken_mask nodes less than the defined graphcutoff, it will be retained and be recorded the position in node feature list (like the relevant function defined in GeoPPI)
            # after getting all residue nodes to be retained, we can use the subgraph function defined below the get the new compact graph for following calculation
            # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/subgraph.html

            wt_effective_reslist = np.array(wt_effective_reslist)
            # mutation_mask = torch.as_tensor(np.where(np.isin(wt_effective_reslist, mutation_crtoken))[0], device=self.device, dtype=torch.int32)
            # interface_mask = torch.as_tensor(np.where(np.isin(wt_effective_reslist, interface_res))[0], device=self.device, dtype=torch.int32)
            # assuming after pre-processing, the cr_token is unique for each residue in a protein (thus np.isin is used)
            # about np.isin(A, B): check whether A is in B
            mutation_mask = torch.as_tensor(np.isin(wt_effective_reslist, mutation_crtoken), device=self.device, dtype=torch.bool)
            interface_mask = torch.as_tensor(np.isin(wt_effective_reslist, interface_res), device=self.device, dtype=torch.bool)
            ddg = torch.as_tensor(ddg, device=self.device, dtype=torch.float32)
            # for PyG, only variable with the name of edge index will automatically increment based on node numbers of each graph, and for features like initial node embedding, only a simple concatenation will be performed
            # print('mutation_mask:', mutation_mask)
            # print('interface_mask:', interface_mask)
            # print('wt_edge_index:', wt_edge_index)
            # print('mt_edge_index:', mt_edge_index)
            # print(mutation_mask.size(), interface_mask.size()) # torch.Size([615]) torch.Size([615])
            assert mutation_mask.size() == interface_mask.size(), 'the sizes of mutation_mask and interface_mask should be the same'

            if self.whether_spatial_graph:
                data = torch_geometric.data.Data(wt_node_s=wt_node_s, wt_node_v=wt_node_v, wt_edge_s=wt_edge_s, wt_edge_v=wt_edge_v, mt_node_s=mt_node_s, mt_node_v=mt_node_v, mt_edge_s=mt_edge_s, mt_edge_v=mt_edge_v,
                                                 wt_edge_index=wt_edge_index, mt_edge_index=mt_edge_index, mutation_mask=mutation_mask, interface_mask=interface_mask, ddg=ddg, mask_size = mutation_mask.size(0),
                                                 wt_extra_edge_s=wt_extra_edge_s, wt_extra_edge_v=wt_extra_edge_v, mt_extra_edge_s=mt_extra_edge_s, mt_extra_edge_v=mt_extra_edge_v,
                                                 wt_extra_edge_index=wt_extra_edge_index, mt_extra_edge_index=mt_extra_edge_index, name=name)
            else:
                data = torch_geometric.data.Data(wt_node_s=wt_node_s, wt_node_v=wt_node_v, wt_edge_s=wt_edge_s, wt_edge_v=wt_edge_v, mt_node_s=mt_node_s, mt_node_v=mt_node_v, mt_edge_s=mt_edge_s, mt_edge_v=mt_edge_v,
                                                 wt_edge_index=wt_edge_index, mt_edge_index=mt_edge_index, mutation_mask=mutation_mask, interface_mask=interface_mask, ddg=ddg, mask_size = mutation_mask.size(0), name=name)

            return data

    # for screening mutation chain related residues in the interface generated by pymol (the mutation chain is included in the screened pairwise interfaces by pymol)
    # if_info (E_I), chainid (chains including mutations)
    def read_inter_result(self, interface, if_info=None, chainid=None, old2new=None):
        # print(if_info, chainid) # HL_VW ['V']
        if if_info is not None:
            info1 = if_info.split('_')
            pA = info1[0]  # chains of interest for protein A
            pB = info1[1]  # chains of interest for protein B

            # construct a mapping between all chains (of interest) in protein A and all chains in protein B
            mappings = {}
            for a in pA:
                for b in pB:
                    if a not in mappings:
                        mappings[a] = [b]
                    else:
                        mappings[a] += [b]
                    if b not in mappings:
                        mappings[b] = [a]
                    else:
                        mappings[b] += [a]

            # print('mappings:', mappings) # mappings: {'H': ['V', 'W'], 'V': ['H', 'L'], 'W': ['H', 'L'], 'L': ['V', 'W']}
            target_chains = []
            for chainidx in chainid: # chain ids for mutation sites, ['V']
                if chainidx in mappings: # whether mutation chain is in the provided chains of interest
                    target_chains += mappings[chainidx]
            # print('target_chains:', target_chains) # target_chains: ['H', 'L']
            # get all corresponding chains (in another protein) of all the mutation chain (in current protein)

            target_inters = []
            for chainidx in chainid:
                target_inters += ['{}_{}'.format(chainidx, y) for y in target_chains] + ['{}_{}'.format(y, chainidx) for y in target_chains] # for the case that y and chainidx are put in different order by pymol
            # print(target_inters) # ['V_H', 'V_L', 'H_V', 'L_V']
            # get all combinations of the mutation chains and corresponding chains (for screening defined interface residues from all interfaces geneerated by pymol (which may not include mutation chains))

            target_inters = list(set(target_inters))
            # print(target_inters) # ['V_L', 'V_H', 'L_V', 'H_V']

        # if partner information is empty
        else:
            target_inters = None

        # open generated interface file (by pymol)
        interlines = interface
        interface_res = []
        for line in interlines: # iterate all pymol interfaces
            iden = line[:3]
            # print(line, iden) # H_L_H_37 H_L

            # only consider the case that target_inters is not empty (interface_res is generated following this rule)
            if target_inters is None:
                if iden.split('_')[0] not in chainid and iden.split('_')[1] not in chainid: # chainid: mutation chain ids
                    continue
                # else: retain pymol interface entries that include mutation chains but are not considered by target_inters
                # I guess that it is for some cases that provided pdb structures not just have chains of interest in original csv mutation file (but these outlier chains still have interfaces with mutation chains)
            else:
                if iden not in target_inters:
                    continue

            infor = line[4:].strip().split('_')  # chainid, resid
            assert len(infor) == 2
            # adding interface position（based on chain id + residue id in this chain）
            interface_res.append('_'.join(infor))

        if old2new is not None:
            mapps = {x[:-4]: y[:-4] for x, y in old2new.items()}
            interface_res = [mapps[x] for x in interface_res if x in mapps]

        return interface_res

    def feature_generator(self, protein):
        name = protein['name']
        seq = protein['seq'] # based on effective/natural AA sequences
        original_coords = protein['coords']
        original_sidec = copy.deepcopy(protein['sidechain_dict'])
        # residue_list records the cr_token sequentially following the residue order of original pdb file
        residue_list = protein['res_idx_list']
        residue_num = len(residue_list)
        assert residue_num > 0, 'Residue number in current protein should be larger than 0: {}'.format(name)

        sidec_atom_list, atom_set = dict(), set()
        # record side chain atom type (following the residue_list order) and atom weight
        for res in residue_list:
            # this residue (identified by cr_token) has the side chain atoms
            if res in original_sidec.keys():
                # currently atoms in side chain are ordered by 'sorted' function
                atoms = [i[0] for i in np.array(original_sidec[res])[:, 0]]
                # self.atom_mass = {'C': 0.1621, 'N': 0.1891, 'O': 0.2160, 'S': 0.4328} only has these four atoms, if other atoms occur, an error will be raised
                sidec_atom_list[res] = torch.as_tensor([self.atom_mass[j] for j in atoms], device=self.device, dtype=torch.float32).view(-1, 1)
                for atom in atoms:
                    atom_set.add(atom)

        # transform coordinate data into tensor
        for key in original_sidec.keys():
            original_sidec[key] = torch.as_tensor(np.array(original_sidec[key])[:, 1:].astype(np.float32), device=self.device, dtype=torch.float32)
        original_coords = torch.as_tensor(original_coords, device=self.device, dtype=torch.float32)

        # currently, the centroid of current protein is based on the coordinates of all backbone atoms (N, CA, C, O) rather than just based on CA
        original_centroid = self.centroid(original_coords)
        original_coords = original_coords - original_centroid # normalize backbone coordinates
        # for the case of NaN value occurring in atom coordinates (currently try to arise errors)
        mask_nan = torch.isfinite(original_coords.sum(dim=(1, 2)))
        assert (~mask_nan).sum() == 0, 'Current pdb has invalid coordinates.'
        original_X_ca = original_coords[:, 1]
        for key in original_sidec.keys():
            original_sidec[key] = original_sidec[key] - original_centroid # normalize side chain coordinates

        # start to generate residue features based on normalized coordinates
        original_edge_index = torch_cluster.knn_graph(original_X_ca, k=self.top_k)  # knn_graph self loop default: False, the self-loop-like operation is realized in GVPConvLayer (formulas 4-5)
        # edge features (3)
        original_pos_embeddings = self._positional_embeddings(original_edge_index)
        original_E_vectors = original_X_ca[original_edge_index[0]] - original_X_ca[original_edge_index[1]]
        original_rbf = _rbf(original_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
        # if the radius graph is needed
        if self.whether_spatial_graph:
            extra_edge_index = torch_cluster.radius_graph(original_X_ca, r=10.0)
            extra_pos_embeddings = self._positional_embeddings(extra_edge_index)
            extra_E_vectors = original_X_ca[extra_edge_index[0]] - original_X_ca[extra_edge_index[1]]
            extra_rbf = _rbf(extra_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

        # node features
        # the below four type of features are the features specified in original GVP-GNN paper
        original_dihedrals = self._dihedrals(original_coords)
        original_orientations = self._orientations(original_X_ca)  # torch.Size([126, 2, 3])
        original_sidechains = self._sidechains(original_coords)
        # here non-natural residues have been removed in _3_generate_json_input.py (no mutation occurs in non-natural AA sites and all mutations are natural AA mutations)
        # in other words, the non-natural AA sites will be ignored in WT and MT simultaneously
        aatype_onehot = F.one_hot(torch.LongTensor([self.letter_to_num[i] for i in seq if i != ':']), num_classes=20).to(self.device)

        # extra node feature
        # which chain current residue locates to
        chainid_seq = []
        for i in range(len(seq.split(':'))): # seq: based on effective/natural AA sequences
            chainid_seq.extend([i] * len(seq.split(':')[i]))
        chainid_seq = torch.as_tensor(chainid_seq, device=self.device, dtype=torch.int32) / i  # dividing i is for normalization

        # residue_list records the cr_token sequentially following the residue order of original pdb file
        # currently the below four four chain coordinate information is not normalized and the vacant places are filled with zero
        if self.sidec_chain_normalization:
            # current centroid of protein backbone complex is calculated based on original_coords (C+CA+N+O), thus the further normalization is also based on this
            # currently original_coords has been through centroid normalization
            length = torch.sqrt(torch.sum((original_coords.view(-1, 3) ** 2), -1))
            length = length[torch.argmax(length)]
            sidec_seq_max = (torch.concat([torch.max(original_sidec[i], dim=0, keepdim=True)[0] if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length)\
                .type(torch.float32).to(self.device)
            sidec_seq_centroid = (torch.concat([torch.mean(original_sidec[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length)\
                .type(torch.float32).to(self.device)
            sidec_seq_mass = (torch.concat([torch.mean(original_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length)\
                .type(torch.float32).to(self.device)
            sidec_CA_relations = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list) # relative value, not influenced by the further normlization
            original_X_ca = original_X_ca / length
        else:
            sidec_seq_max = torch.concat(
                [torch.max(original_sidec[i], dim=0, keepdim=True)[0] if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]
            ).type(torch.float32).to(self.device)
            sidec_seq_centroid = torch.concat(
                [torch.mean(original_sidec[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]
            ).type(torch.float32).to(self.device)
            sidec_seq_mass = torch.concat( # the below i is the cr_token key in res_idx_list
                [torch.mean(original_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list] # also based on original_centroid
            ).type(torch.float32).to(self.device)
            sidec_CA_relations = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list)
        assert aatype_onehot.size(0)== original_dihedrals.size(0) == original_orientations.size(0) == original_sidechains.size(0) == sidec_seq_max.size(0) == sidec_seq_centroid.size(0) \
               == sidec_seq_mass.size(0) == sidec_CA_relations.size(0) == chainid_seq.size(0) == original_X_ca.size(0), 'All generated features does not have the same length.'

        # extra CG features
        if self.whether_CG_feature:
            # cg_feature is based on res_idx_list, i.e., residue_list here, same to the retrieve of the backbone coordinates
            cg_feature = protein['cg_feature']
            pos_feats_num = torch.as_tensor([9, 12, 6, 3, 2], device=self.device, dtype=torch.int64)  # feature number for each position of cg features (5 pos in total)
            if self.CG_encoding_type == 'sincos':
                # normalize each column/feature separately using pos_feats_num
                cg_feature = (torch.as_tensor(cg_feature, device=self.device, dtype=torch.float32) / pos_feats_num) * 6.283
                cg_feature = torch.cat([torch.sin(cg_feature), torch.cos(cg_feature)], dim=-1)
            elif self.CG_encoding_type == 'onehot':
                # old version:
                # cg_feature = F.one_hot(torch.LongTensor(cg_feature), num_classes=18).to(self.device)
                # cg_feature = cg_feature.view(cg_feature.size(0), -1)
                # new version:
                cg_feature = torch.LongTensor(cg_feature).to(self.device) # F.one_hot only takes int64 as the input
                cg_feature = torch.cat([F.one_hot(cg_feature[:, i], num_classes=pos_feats_num[i]) for i in range(pos_feats_num.size(0))], dim=-1).float()
            else:
                cg_feature = torch.as_tensor(cg_feature, device=self.device, dtype=torch.float32) / pos_feats_num

            assert aatype_onehot.size(0) == cg_feature.size(0)
            node_s = torch.cat([aatype_onehot, original_dihedrals, chainid_seq.unsqueeze(-1), cg_feature], dim=-1)
        else:
            node_s = torch.cat([aatype_onehot, original_dihedrals, chainid_seq.unsqueeze(-1)], dim=-1)

        node_v = torch.cat([original_orientations, original_sidechains.unsqueeze(-2), sidec_seq_max.unsqueeze(-2), sidec_seq_centroid.unsqueeze(-2), sidec_seq_mass.unsqueeze(-2), sidec_CA_relations, original_X_ca.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([original_rbf, original_pos_embeddings], dim=-1)
        edge_v = _normalize(original_E_vectors).unsqueeze(-2)

        if self.whether_spatial_graph:
            extra_edge_s = torch.cat([extra_rbf, extra_pos_embeddings], dim=-1)
            extra_edge_v = _normalize(extra_E_vectors).unsqueeze(-2)
            node_s, node_v, edge_s, edge_v, extra_edge_s, extra_edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v, extra_edge_s, extra_edge_v))
            return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list, extra_edge_index, extra_edge_s, extra_edge_v
        else:
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
            return node_s, node_v, edge_s, edge_v, original_edge_index, residue_list

    def centroid(self, coords):
        """
        Centroid is the mean position of all the points in all of the coordinate
        directions, from a vectorset X.
        https://en.wikipedia.org/wiki/Centroid
        C = sum(X)/len(X)
        Parameters
        ----------
        X : array
            (N,D) matrix, where N is points and D is dimension.
        Returns
        """
        coords = coords.view(-1, 3).mean(axis=0)
        return coords

    def _positional_embeddings(self, edge_index, num_embeddings=None, period_range=[2, 1000]):

        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings  # 16
        d = edge_index[0] - edge_index[1]
        # print(d.size()) torch.Size([11880])

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        # print(frequency.size()) torch.Size([8])
        angles = d.unsqueeze(-1) * frequency
        # print(angles.size()) torch.Size([11880, 8])
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        # print(E.size()) torch.Size([11880, 16])
        return E

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        # X: torch.Size([166, 4, 3]) # backbone atom coordinates for current protein
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        # print(X.size()) torch.Size([534, 3]) # remove O coordinates, and flatten the rest of atom coordinates (N，Ca，C)
        dX = X[1:] - X[:-1] # Ca1, C1, N2, Ca2, C2, N3, Ca3, C3 # N1, Ca1, C1, N2, Ca2, C2, N3, Ca3

        # print(X.size(), dX.size()) torch.Size([330, 3]) torch.Size([329, 3])
        U = _normalize(dX, dim=-1)

        u_2 = U[:-2] # these operations are used in the sample dimension
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1) # calculate the cross product, and then do the normalization for it
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        # sign函数（-1-0-1）（感觉用处主要是为后面返回的弧度区分正负（根据在化学中使用二面角的定义），https://en.wikipedia.org/wiki/Dihedral_angle），acos：computes the inverse cosine of each element in input（返回的是弧度）
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
        # print(D.size()) torch.Size([552])

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) # 相当于在生成的一维D（各二面角）的基础上在最前面填一个0，在最后填两个0，感觉就是在整个计算过程中，对于一个蛋白质，计算其中各氨基酸的二面角时，缺少了phi[0], psi[-1], omega[-1]，然后使用0进行填补
        # print(D.size(), D) torch.Size([555])

        D = torch.reshape(D, [-1, 3]) # 将φi, ψi, ωi的特征赋给每一个氨基酸（在当前scheme下，所有蛋白质的第一个氨基酸和最后的一个氨基酸会一共缺失三个特征：phi[0], psi[-1], omega[-1]）
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        # print(D_features.size(), X.size()) # torch.Size([330, 6]) torch.Size([990, 3]) # 应该是为每一个蛋白质的氨基酸分配一个维度为6的特征
        return D_features

    def _orientations(self, X):
        # print(X[1:].size(), X[:-1].size()) # torch.Size([125, 3]) torch.Size([125, 3])
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        # print(forward.size())
        # print(backward.size())
        forward = F.pad(forward, [0, 0, 0, 1])  # [274, 3]，在最下面填一层0，因为感觉第一个值没有计算
        backward = F.pad(backward, [0, 0, 1, 0])  # [274, 3]，在最上面填一层0，最后一个值没有计算
        # print(forward.size())  # [275, 3]
        # print(backward.size())  # [275, 3]
        # print(X.size()) # [275, 3]
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        # torch.Size([166, 4, 3]) (for N, Ca, C, O)
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec

    # currently _CA_to_sidechains is based on normalized CA coordinates instead of the absolute ones
    # sidec_CA_relations = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list)
    def _CA_to_sidechains(self, CA, sidec, residue_list):
        assert CA.size(0) == len(residue_list) # one CA corresponds to one residue_list element
        vec = []
        for i in range(len(residue_list)):
            if residue_list[i] in sidec.keys():
                forward = torch.mean(_normalize(sidec[residue_list[i]] - CA[i]), dim=0, keepdim=True)
                backward = torch.mean(_normalize(CA[i] - sidec[residue_list[i]]), dim=0, keepdim=True)
                overall = torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)
                vec.append(overall)
            else:
                overall = torch.zeros(1, 2, 3).to(self.device)
                vec.append(overall)

        return torch.cat(vec, dim=0)

