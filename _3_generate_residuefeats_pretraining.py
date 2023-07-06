# step3:
# based on pdbs with completed side chains, generate pytorch DataLoader for pretraining
import json
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import copy


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


# this is data source for pretraining, rather the formal pytorch Dataset sent to pytorch DataLoader
# thus here is only for loading basic data without any extra feature calculation, feature calculation is performed in pytorch Dataset
class PretrainingDataset:
    def __init__(self, path, splits_path):
        with open(splits_path) as f:
            # The json. load() is used to read the JSON document from file and The json. loads() is used to convert the JSON String document into the Python dictionary.
            dataset_splits = json.load(f)
            # print(dataset_splits['train'], dataset_splits['val'])

        # for pre-training dataset, only training and validation are needed
        train_list, val_list = dataset_splits['train'], dataset_splits['val']
        self.train, self.val = [], []

        # read original data that has been processed by jupyter file, to form information in the format of GVP-GNN
        with open(path) as f:
            lines = f.readlines() # each entry in lines is a json txt file waiting to be transformed into python dict

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            # get required information for gvp training here

            name = entry['name']
            coords = entry['coords']
            # here is only for coordinate format adjustment and dataset split
            entry['coords'] = list(zip(coords['N'], coords['CA'], coords['C'], coords['O']))
            # [([3.43, -2.059, 57.593], [4.785, -2.49, 57.148], [4.821, -2.46, 55.629], [5.138, -3.464, 54.967]), ([4.503, -1.273, 55.106], [4.43, -0.953, 53.669], [5.793, -0.89, 52.96], [5.94, -0.201, 51.946])]

            # 'name' does not have '.pdb' suffix
            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)


# after defining source data class, we need to define relevant pytorch Dataset class to produce required node and edge features
# here we need to consider how to add perturbation to coordinates etc. for gvp pre-training as well as surface aware features
class PretrainingGraphDataset(data.Dataset):
    def __init__(self, data_list, noise_type: str, noise: float = 1.0, mask_prob: float = 0.15, only_CA = True, if_sidec_noise = True, SASA_mask = True, num_positional_embeddings=16, top_k = 30, num_rbf=16, device='cpu',
                 sidec_chain_normalization = False, whether_AA_prediction = False, whether_spatial_graph = False, whether_sidec_prediction = False, whether_CG_feature = False, CG_encoding_type='sincos', CG_mask = False):
        super(PretrainingGraphDataset, self).__init__()

        self.data_list = data_list # training/val list that stores each entry in the format of dict
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        # current 'seq' includes 'TER' symbol ':', thus we need to remove it when calculating node_counts
        # self.node_counts_ = [len(e['seq']) for e in data_list]
        self.node_counts = [len(''.join(e['seq'].split(':'))) for e in data_list] # used for passing residue node number to PyG DataLoader
        # print(list(zip([i['name'] for i in data_list], self.node_counts)))

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.atom_mass = {'C': 0.1621, 'N': 0.1891, 'O': 0.2160, 'S': 0.4328}
        # temp = 12.011 + 14.0067 + 15.9994 + 32.06
        # print(12.011/temp, 14.0067/temp, 15.9994/temp, 32.06/temp)

        self.whether_sidec_prediction = whether_sidec_prediction
        self.sidec_chain_normalization = sidec_chain_normalization  # for further normalization of side chain coordinates
        self.whether_AA_prediction = whether_AA_prediction # for adding an extra AA prediction task
        self.whether_spatial_graph = whether_spatial_graph
        self.whether_CG_feature = whether_CG_feature # indicating whether to use CG features
        self.CG_encoding_type = CG_encoding_type  # 'sin'/'onehot'/None

        # about generating noise for node coordinates
        self.noise_type = noise_type
        self.noise = noise # noise is used to control the magnitude of generated noise (similar to control the variance on the top of the standard normal distribution)
        self.mask_prob = mask_prob
        self.only_CA = only_CA # only perturb coordinates of CA in backbone (while the protein centroid is calculated based on C/CA/N/O in backbone), else disturb the whole backbone
        self.if_sidec_noise = if_sidec_noise # whether add noise to side chain atoms
        # print('noise type, magnitude, and possibility:', noise_type, noise, mask_prob)
        self.SASA_mask = SASA_mask # for determining whether to use SASA as an auxiliary prediction task
        self.CG_mask = CG_mask # for determining whether to use CG as an auxiliary prediction task (given self.whether_CG_feature = True)
        self.counter = 0

        if self.noise_type == 'trunc_normal': # draw from normal distribution with truncated value
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 4, 3) * self.noise, a_min=-self.noise * 2.0, a_max=self.noise * 2.0) # in original version of uni-mol, the generated noise is [num_mask, 3] shape, currently is [num_mask, 4, 3] shape
        elif self.noise_type == 'normal':
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 4, 3) * self.noise
        elif self.noise_type == 'uniform':
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 4, 3))
        else:
            self.noise_f = lambda num_mask: 0.0

        if self.if_sidec_noise == True: # currently generate a noise for side chain atom set of each residue
            if self.noise_type == 'trunc_normal':
                self.sidec_noise_f = lambda num_mask: np.clip(
                    np.random.randn(num_mask, 3) * self.noise, a_min=-self.noise * 2.0, a_max=self.noise * 2.0)
            elif self.noise_type == 'normal':
                self.sidec_noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
            elif self.noise_type == 'uniform':
                self.sidec_noise_f = lambda num_mask: np.random.uniform(
                    low=-self.noise, high=self.noise, size=(num_mask, 3))
            else:
                self.sidec_noise_f = lambda num_mask: 0.0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self._featurize_as_graph(self.data_list[i], i)

    # *** need to check whether the generated mask and random coordinate perturbation will change in different epochs ***
    # *** need to ensure no operation will influence the original data stored in self.data_list (protein in _featurize_as_graph in originated from self.data_list[i]) ***
    # *** otherwise the data generated for the next batch will be changed, causing errors ***
    def _featurize_as_graph(self, protein, i):
        # self.counter += 1
        name = protein['name']
        # get complete entry of protein
        with torch.no_grad(): # close the gradiant trace
            seq = protein['seq']
            sasa_dict = protein['sasa_dict']
            # according to current logic, we need to corrupt coordinates at first, and generate features based on corrupted coordinates
            original_coords = protein['coords']
            # for solving the above problem, if we do not add deepcopy data_list[i]['sidechain_dict'] will change below
            # we need notice that the below way to process original_sidec will change the original self.data_list
            # original_sidec[key] = torch.as_tensor(np.array(original_sidec[key])[:, 1:].astype(np.float32), device=self.device, dtype=torch.float32)
            original_sidec = copy.deepcopy(protein['sidechain_dict'])

            # residue_list records the cr_token sequentially following the residue order of original pdb file
            residue_list = protein['res_idx_list']
            residue_num = len(residue_list) # protein['res_idx_list'] stores all residue identifier in current protein
            assert residue_num > 0, 'Residue number in current protein should be larger than 0: {}'.format(name)
            sidec_atom_list, atom_set = dict(), set()
            # record side chain atom type (following the residue_list order) and atom weight
            for res in residue_list:
                if res in original_sidec.keys():
                    # only retain the main atom name, like C/S/O/N
                    atoms = [i[0] for i in np.array(original_sidec[res])[:, 0]]
                    # atoms = []
                    # for i in np.array(original_sidec[res])[:, 0]:
                    #     atoms.append(i[0])
                    sidec_atom_list[res] = torch.as_tensor([self.atom_mass[j] for j in atoms], device=self.device, dtype=torch.float32).view(-1, 1)
                    for atom in atoms:
                        atom_set.add(atom)
            # print(atom_set) # {'C', 'S', 'O', 'N'} only C, S, O, N exist in side chain atoms of the pretraining set
            # record side chain atom coordinate tensor
            for key in original_sidec.keys():
                original_sidec[key] = torch.as_tensor(np.array(original_sidec[key])[:, 1:].astype(np.float32), device=self.device, dtype=torch.float32)

            # 1. start to generate residue mask
            # np.random.rand(): Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
            num_mask = int(self.mask_prob * residue_num + np.random.rand())
            mask_res = np.random.choice(residue_list, num_mask, replace=False).tolist() # replace=False: no duplicates, the order in mask_res is perturbed
            # print('mask_res:', len(mask_res), mask_res)
            mask = np.isin(residue_list, mask_res) # a numpy mask determining whether elements in A occur in B (np.isin(A, B)), in the order of residue_list
            # print('protein.keys():', protein.keys()) # dict_keys(['seq', 'coords', 'num_chains', 'name', 'sidechain_dict', 'sasa_dict', 'res_idx_list'])
            # start from here, we need to generate the corrupted residue coordinates (the masked residue positions are recorded by 'mask')
            # currently follow the scheme that, generating all node and edge features after corrupting coordinates
            num_mask = mask.astype(np.int32).sum() # boolean to int, mask value = True/1 represents this residue will be corrupted

            # 2. start to process coordinates, all features generated will be based on new coordinates
            new_coords = np.copy(np.array(original_coords)) # (126, 4, 3)
            if self.only_CA == True:
                temp_noise = self.noise_f(num_mask)[:, 1]
                new_coords[mask, 1] += temp_noise
                # print('current noise:', temp_noise[0]) # print the noise generated for the first residue of current protein
            else:
                temp_noise = self.noise_f(num_mask)
                new_coords[mask, :] += temp_noise

            # all coordinates in new_coords should be retained, as we need to the whole new_coords to generate residue graph and features
            # and then retrieve the corrupted residues through mask
            original_coords = torch.as_tensor(original_coords, device=self.device, dtype=torch.float32)
            new_coords = torch.as_tensor(new_coords, device=self.device, dtype=torch.float32)

            # because for the CA denoising task, what gvp predicts is the relative position between absolute original coordinates and absolute perturbed coordinates
            # thus we need to get absolute CA coordinates here before coordinate normalization as the labels
            # the below generated original_X_ca and new_X_ca are based on normalized coordinates for calculating other labels/features, like side chain geometric information
            original_ca_label = original_coords[:, 1][mask] # atom order here: N, CA, C, O, based on absolute coordinates
            new_ca_label = new_coords[:, 1][mask]

            # currently, the centroid of current protein is based on the coordinates of all backbone atoms (N, CA, C, O) rather than just based on CA
            original_centroid = self.centroid(original_coords)
            new_centroid = self.centroid(new_coords)
            original_coords = original_coords - original_centroid
            new_coords = new_coords - new_centroid # this is the normalization of new backbone coordinates, next, we also need to normalize side chain atoms using the centroid

            # for the case of NaN value occurring in atom coordinates (currently try to arise errors)
            mask_nan = torch.isfinite(new_coords.sum(dim=(1, 2)))
            assert (~mask_nan).sum() == 0, 'Current pdb has invalid coordinates.' # if new_coords has invalid coordinates, original_coords will have them as well
            original_X_ca = original_coords[:, 1] # based on normalized coordinates, to generate other relevant features
            new_X_ca = new_coords[:, 1]

            # need to consider the correspondence between corrupted side chain coordinates and the ground truth coordinates
            # current solution: after deepcopy, for each residue, its side chain atom order will not change
            new_sidec = copy.deepcopy(original_sidec)
            for key in original_sidec.keys():
                original_sidec[key] = original_sidec[key] - original_centroid
            for key in new_sidec.keys(): # side chain coordinate normalization
                new_sidec[key] = new_sidec[key] - new_centroid

            # *** actually, if we add noise to residue coordinates and generate side chain atom feature based on the relative position between residue and side chain coordinates ***
            # *** in this case, maybe it is also feasible to just add noise to the residue rather than adding noise to side chain atom as well, because the relative position will change simultaneously when residue noise is added ***
            # *** furthermore, the denoising process includes residue coordinate denoising and the side chain relative position denoising (sub-tasks) ***
            # *** if set if_sidec_noise to true, the relevant residue node embedding will change, in other words, even though the corresponding loss is not calculated, it will also influence the prediction of other tasks ***
            if self.if_sidec_noise:
                sorted_mask_res = sorted(mask_res)
                assert num_mask == len(sorted_mask_res), 'relevant error pdb name: {}, num_mask: {}, len(sorted_mask_res): {}'.format(name, num_mask, len(sorted_mask_res))
                sidec_noise_f = torch.as_tensor(self.sidec_noise_f(num_mask), device=self.device, dtype=torch.float32)
                for i in range(num_mask):
                    # need to consider one case that the side chain atom coordinates to be perturbed do not exist in current pdb
                    if sorted_mask_res[i] in new_sidec.keys():
                        # previous logic: add the same noise to every atom coordinate in the side chain
                        # new_sidec[sorted_mask_res[i]] = new_sidec[sorted_mask_res[i]] + sidec_noise_f[i]
                        # new logic: randomly select one side chain atom and add noise that so that the centroid of side chain atom set could be broken and be denoising later
                        random_index = random.sample(range(new_sidec[sorted_mask_res[i]].size(0)), 1)
                        new_sidec[sorted_mask_res[i]][random_index] = new_sidec[sorted_mask_res[i]][random_index] + sidec_noise_f[i]

            # 3. start to generate features based on normalized coordinates
            # torch_cluster also includes radius graph
            new_edge_index = torch_cluster.knn_graph(new_X_ca, k=self.top_k) # knn_graph self loop default: False, the self-loop-like operation is realized in GVPConvLayer (formulas 4-5)
            # these features are generated based on corrupted coordinates
            # edge features (3)
            new_pos_embeddings = self._positional_embeddings(new_edge_index)
            new_E_vectors = new_X_ca[new_edge_index[0]] - new_X_ca[new_edge_index[1]]
            new_rbf = _rbf(new_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            # if the radius graph is needed
            if self.whether_spatial_graph:
                extra_edge_index = torch_cluster.radius_graph(new_X_ca, r=10.0)
                extra_pos_embeddings = self._positional_embeddings(extra_edge_index)
                extra_E_vectors = new_X_ca[extra_edge_index[0]] - new_X_ca[extra_edge_index[1]]
                extra_rbf = _rbf(extra_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            # node features
            # the below four type of features are the features specified in original GVP-GNN paper
            new_dihedrals = self._dihedrals(new_coords)
            new_orientations = self._orientations(new_X_ca) # torch.Size([126, 2, 3])
            new_sidechains = self._sidechains(new_coords)
            aatype_onehot = F.one_hot(torch.LongTensor([self.letter_to_num[i] for i in seq if i!=':']), num_classes = 20).to(self.device)

            # extra node feature
            # SASA, need to check the numerical range of SASA (currently choose not to normalize SASA)
            # original SASA file: ResidueID, ChainID, ASA_complex, ASA_single, dASA, Interface, currently SASA used in pretraining is asa_complex
            SASA = torch.as_tensor([float(sasa_dict[i]['asa_complex']) for i in residue_list], device=self.device, dtype=torch.float32) # every residue will be equipped with a SASA and an interface value
            # whether in complex interface or not
            interface = torch.as_tensor([int(sasa_dict[i]['interface']) for i in residue_list], device=self.device, dtype=torch.int32)
            # which chain current residue locates to
            chainid_seq = []
            for i in range(len(seq.split(':'))):
                chainid_seq.extend([i] * len(seq.split(':')[i]))
            chainid_seq = torch.as_tensor(chainid_seq, device=self.device, dtype=torch.int32)/i # dividing i is for normalization
            assert SASA.size() == interface.size() == chainid_seq.size(), 'SASA, interface, chainid_seq should have the same size in {}.'.format(name)

            # residue_list records the cr_token sequentially following the residue order of original pdb file
            # currently the below four four chain coordinate information has not been further normalized and the vacant places are filled with zero
            if self.sidec_chain_normalization:
                # current centroid of protein backbone complex is calculated based on original_coords (C+CA+N+O), thus the further normalization is also based on this
                # currently original_coords has been through centroid normalization
                length = torch.sqrt(torch.sum((new_coords.view(-1, 3) ** 2), -1))
                length = length[torch.argmax(length)]
                sidec_seq_max = (torch.concat([torch.max(new_sidec[i], dim=0, keepdim=True)[0] if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                    .type(torch.float32).to(self.device)
                sidec_seq_centroid = (torch.concat([torch.mean(new_sidec[i], dim=0, keepdim=True) if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                    .type(torch.float32).to(self.device)
                sidec_seq_mass = (torch.concat([torch.mean(new_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                    .type(torch.float32).to(self.device)
                sidec_CA_relations = self._CA_to_sidechains(new_X_ca, new_sidec, residue_list)  # relative value, not influenced by the further normlization
                new_X_ca = new_X_ca / length
            else:
                sidec_seq_max = torch.concat([torch.max(new_sidec[i], dim=0, keepdim=True)[0] if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                    .type(torch.float32).to(self.device)
                sidec_seq_centroid = torch.concat([torch.mean(new_sidec[i], dim=0, keepdim=True) if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                    .type(torch.float32).to(self.device)
                sidec_seq_mass = torch.concat([torch.mean(new_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                    .type(torch.float32).to(self.device) # also based on new_centroid
                # print(new_sidec['A_1'].size(), sidec_atom_list['A_1'].size(), sidec_seq_mass.size())
                sidec_CA_relations = self._CA_to_sidechains(new_X_ca, new_sidec, residue_list)

            # currently no surface aware features are provided here, because these features are more useful in protein-protein docking rather than here
            # https://github.com/octavian-ganea/equidock_public/blob/main/src/utils/protein_utils.py
            # current node feature overall
            # 1. aa type 2. SASA (objective) 3. three CA relevant features in GVP-GNN 4. whether on interface or not
            # 5. four side chain features (objective, vector) 6. side chain ids 7. normalized CA coordinates (objective: predicting relative distance between ground truth and perturbation coordinates, vector)
            assert aatype_onehot.size(0) == SASA.size(0) == new_dihedrals.size(0) == new_orientations.size(0) == new_sidechains.size(0) == interface.size(0)\
                   == sidec_seq_max.size(0) == sidec_seq_centroid.size(0) == sidec_seq_mass.size(0) == sidec_CA_relations.size(0) == chainid_seq.size(0) == new_X_ca.size(0), 'All generated features does not have the same length.'

            # 4. start to generate labels for each sub prediction tasks
            # SASA (mask the positions to be predicted)
            if self.SASA_mask == True:
                SASA_label = SASA[mask].unsqueeze(-1)
                SASA[mask] = 0
            else:
                # in both cases, SASA_label is generated for other potential use
                # while in this case, since SASA prediction is not treated as an auxiliary task, the corresponding SASA value in residues will be retained
                SASA_label = SASA[mask].unsqueeze(-1)

            # we need calculate the original corresponding features
            if self.whether_sidec_prediction == True:
                if self.sidec_chain_normalization:
                    length = torch.sqrt(torch.sum((original_coords.view(-1, 3) ** 2), -1))
                    length = length[torch.argmax(length)]
                    sidec_seq_max_label = (torch.concat([torch.max(original_sidec[i], dim=0, keepdim=True)[0] if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_seq_centroid_label = (torch.concat([torch.mean(original_sidec[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_seq_mass_label = (torch.concat([torch.mean(original_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_CA_relations_label = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list)[mask]  # relative value, not influenced by the further normlization
                else:
                    sidec_seq_max_label = torch.concat([torch.max(original_sidec[i], dim=0, keepdim=True)[0] if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_seq_centroid_label = torch.concat([torch.mean(original_sidec[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_seq_mass_label = torch.concat([torch.mean(original_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                        .type(torch.float32).to(self.device)[mask] # also based on new_centroid
                    # currently _CA_to_sidechains is based on normalized CA coordinates instead of the absolute ones
                    sidec_CA_relations_label = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list)[mask]
                sidec_label = torch.cat([sidec_seq_max_label.unsqueeze(-2), sidec_seq_centroid_label.unsqueeze(-2), sidec_seq_mass_label.unsqueeze(-2), sidec_CA_relations_label], dim=-2)
            else:
                sidec_label = torch.zeros(SASA_label.size(0), 4).to(self.device)

            # extra AA prediction label
            if self.whether_AA_prediction:
                AA_prediction_label = aatype_onehot[mask].float()
                aatype_onehot[mask] = 0
            else:
                AA_prediction_label = aatype_onehot[mask].float()

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
                    cg_feature = torch.LongTensor(cg_feature).to(self.device)
                    cg_feature = torch.cat([F.one_hot(cg_feature[:, i], num_classes=pos_feats_num[i]) for i in range(pos_feats_num.size(0))], dim=-1).float()
                else:
                    cg_feature = torch.as_tensor(cg_feature, device=self.device, dtype=torch.float32) / pos_feats_num
                assert aatype_onehot.size(0) == cg_feature.size(0)  # check for the size of cg_feature

                if self.CG_mask == True:
                    CG_label = cg_feature[mask]
                    cg_feature[mask] = 0
                # merge all information and return it
                node_s = torch.cat([aatype_onehot, SASA.unsqueeze(-1), new_dihedrals, interface.unsqueeze(-1), chainid_seq.unsqueeze(-1), cg_feature], dim=-1)
            else:
                # print(aatype_onehot.size(), SASA.size(), new_dihedrals.size(), new_orientations.size(), new_sidechains.size(), interface.size(),
                #       sidec_seq_max.size(), sidec_seq_centroid.size(), sidec_seq_mass.size(), sidec_CA_relations.size(), chainid_seq.size(), new_X_ca.size())
                # torch.Size([126, 20]) torch.Size([126]) torch.Size([126, 6]) torch.Size([126, 2, 3]) torch.Size([126, 3]) torch.Size([126])
                # torch.Size([126, 3]) torch.Size([126, 3]) torch.Size([126, 3]) torch.Size([126, 2, 3]) torch.Size([126]) torch.Size([126, 3])
                node_s = torch.cat([aatype_onehot, SASA.unsqueeze(-1), new_dihedrals, interface.unsqueeze(-1), chainid_seq.unsqueeze(-1)], dim=-1)

            node_v = torch.cat([new_orientations, new_sidechains.unsqueeze(-2), sidec_seq_max.unsqueeze(-2), sidec_seq_centroid.unsqueeze(-2), sidec_seq_mass.unsqueeze(-2), sidec_CA_relations, new_X_ca.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([new_rbf, new_pos_embeddings], dim=-1)
            edge_v = _normalize(new_E_vectors).unsqueeze(-2)
            mask = torch.as_tensor(mask, device=self.device) # transform to tensor for letting PyG Data to stack masks of multiple proteins (as a complete tensor) correctly

            if self.whether_spatial_graph:
                extra_edge_s = torch.cat([extra_rbf, extra_pos_embeddings], dim=-1)
                extra_edge_v = _normalize(extra_E_vectors).unsqueeze(-2)
                if self.whether_CG_feature and self.CG_mask:
                    node_s, node_v, edge_s, edge_v, SASA_label, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v, CG_label = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, SASA_label, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v, CG_label))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, SASA_label=SASA_label,
                                                     sidec_label=sidec_label, original_ca_label=original_ca_label, new_ca_label=new_ca_label, AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask,
                                                     extra_edge_index=extra_edge_index, extra_edge_s=extra_edge_s, extra_edge_v=extra_edge_v, CG_label=CG_label)
                else:
                    node_s, node_v, edge_s, edge_v, SASA_label, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, SASA_label, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, SASA_label=SASA_label,
                                                     sidec_label=sidec_label, original_ca_label=original_ca_label, new_ca_label=new_ca_label, AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask,
                                                     extra_edge_index=extra_edge_index, extra_edge_s=extra_edge_s, extra_edge_v=extra_edge_v)

            else:
                if self.whether_CG_feature and self.CG_mask:
                    node_s, node_v, edge_s, edge_v, SASA_label, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, CG_label = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, SASA_label, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, CG_label))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, SASA_label=SASA_label, sidec_label=sidec_label,
                                                     original_ca_label=original_ca_label, new_ca_label=new_ca_label, AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask, CG_label=CG_label)
                else:
                    node_s, node_v, edge_s, edge_v, SASA_label, sidec_label, original_ca_label, new_ca_label, AA_prediction_label = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, SASA_label, sidec_label, original_ca_label, new_ca_label, AA_prediction_label))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, SASA_label=SASA_label,
                                                     sidec_label=sidec_label, original_ca_label=original_ca_label, new_ca_label=new_ca_label, AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask)
                    # Data(edge_index=[2, 3780], name='1a0a_1', node_s=[126, 29], node_v=[126, 9, 3], edge_s=[3780, 32], edge_v=[3780, 1, 3], SASA_label=[19], sidec_label=[19, 5, 3], X_ca_label=[19, 3], mask=[126])

            return data

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


class PretrainingGraphDataset_wosidec(data.Dataset):
    def __init__(self, data_list, noise_type: str, noise: float = 1.0, mask_prob: float = 0.15, only_CA = True, SASA_mask = True, num_positional_embeddings=16, top_k = 30, num_rbf=16, device='cpu',
                 whether_AA_prediction = False, whether_spatial_graph = False, remove_SASA_all = False):
        super(PretrainingGraphDataset_wosidec, self).__init__()

        self.data_list = data_list # training/val list that stores each entry in the format of dict
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        # current 'seq' includes 'TER' symbol ':', thus we need to remove it when calculating node_counts
        # self.node_counts_ = [len(e['seq']) for e in data_list]
        self.node_counts = [len(''.join(e['seq'].split(':'))) for e in data_list] # used for passing residue node number to PyG DataLoader
        # print(list(zip([i['name'] for i in data_list], self.node_counts)))

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.atom_mass = {'C': 0.1621, 'N': 0.1891, 'O': 0.2160, 'S': 0.4328}
        # temp = 12.011 + 14.0067 + 15.9994 + 32.06
        # print(12.011/temp, 14.0067/temp, 15.9994/temp, 32.06/temp)

        self.whether_AA_prediction = whether_AA_prediction # for adding an extra AA prediction task
        self.whether_spatial_graph = whether_spatial_graph
        self.remove_SASA_all = remove_SASA_all

        # about generating noise for node coordinates
        self.noise_type = noise_type
        self.noise = noise # noise is used to control the magnitude of generated noise (similar to control the variance on the top of the standard normal distribution)
        self.mask_prob = mask_prob
        self.only_CA = only_CA # only perturb coordinates of CA in backbone (while the protein centroid is calculated based on C/CA/N/O in backbone), else disturb the whole backbone
        # print('noise type, magnitude, and possibility:', noise_type, noise, mask_prob)
        self.SASA_mask = SASA_mask # for determining whether to use SASA as an auxiliary prediction task
        self.counter = 0

        if self.noise_type == 'trunc_normal': # draw from normal distribution with truncated value
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 4, 3) * self.noise, a_min=-self.noise * 2.0, a_max=self.noise * 2.0) # in original version of uni-mol, the generated noise is [num_mask, 3] shape, currently is [num_mask, 4, 3] shape
        elif self.noise_type == 'normal':
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 4, 3) * self.noise
        elif self.noise_type == 'uniform':
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 4, 3))
        else:
            self.noise_f = lambda num_mask: 0.0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self._featurize_as_graph(self.data_list[i], i)

    # *** need to check whether the generated mask and random coordinate perturbation will change in different epochs ***
    # *** need to ensure no operation will influence the original data stored in self.data_list (protein in _featurize_as_graph in originated from self.data_list[i]) ***
    # *** otherwise the data generated for the next batch will be changed, causing errors ***
    def _featurize_as_graph(self, protein, i):
        # self.counter += 1
        name = protein['name']
        # get complete entry of protein
        with torch.no_grad(): # close the gradiant trace
            seq = protein['seq']
            sasa_dict = protein['sasa_dict']
            # according to current logic, we need to corrupt coordinates at first, and generate features based on corrupted coordinates
            original_coords = protein['coords']

            # residue_list records the cr_token sequentially following the residue order of original pdb file
            residue_list = protein['res_idx_list']
            residue_num = len(residue_list) # protein['res_idx_list'] stores all residue identifier in current protein
            assert residue_num > 0, 'Residue number in current protein should be larger than 0: {}'.format(name)

            # 1. start to generate residue mask
            # np.random.rand(): Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
            num_mask = int(self.mask_prob * residue_num + np.random.rand())
            mask_res = np.random.choice(residue_list, num_mask, replace=False).tolist() # replace=False: no duplicates, the order in mask_res is perturbed
            # print('mask_res:', len(mask_res), mask_res)
            mask = np.isin(residue_list, mask_res) # a numpy mask determining whether elements in A occur in B (np.isin(A, B)), in the order of residue_list
            # print('protein.keys():', protein.keys()) # dict_keys(['seq', 'coords', 'num_chains', 'name', 'sidechain_dict', 'sasa_dict', 'res_idx_list'])
            # start from here, we need to generate the corrupted residue coordinates (the masked residue positions are recorded by 'mask')
            # currently follow the scheme that, generating all node and edge features after corrupting coordinates
            num_mask = mask.astype(np.int32).sum() # boolean to int, mask value = True/1 represents this residue will be corrupted

            # 2. start to process coordinates, all features generated will be based on new coordinates
            new_coords = np.copy(np.array(original_coords)) # (126, 4, 3)
            if self.only_CA == True:
                temp_noise = self.noise_f(num_mask)[:, 1]
                new_coords[mask, 1] += temp_noise
                # print('current noise:', temp_noise[0]) # print the noise generated for the first residue of current protein
            else:
                temp_noise = self.noise_f(num_mask)
                new_coords[mask, :] += temp_noise

            # all coordinates in new_coords should be retained, as we need to the whole new_coords to generate residue graph and features
            # and then retrieve the corrupted residues through mask
            original_coords = torch.as_tensor(original_coords, device=self.device, dtype=torch.float32)
            new_coords = torch.as_tensor(new_coords, device=self.device, dtype=torch.float32)

            # because for the CA denoising task, what gvp predicts is the relative position between absolute original coordinates and absolute perturbed coordinates
            # thus we need to get absolute CA coordinates here before coordinate normalization as the labels
            # the below generated original_X_ca and new_X_ca are based on normalized coordinates for calculating other labels/features, like side chain geometric information
            original_ca_label = original_coords[:, 1][mask] # atom order here: N, CA, C, O, based on absolute coordinates
            new_ca_label = new_coords[:, 1][mask]

            # currently, the centroid of current protein is based on the coordinates of all backbone atoms (N, CA, C, O) rather than just based on CA
            original_centroid = self.centroid(original_coords)
            new_centroid = self.centroid(new_coords)
            original_coords = original_coords - original_centroid
            new_coords = new_coords - new_centroid # this is the normalization of new backbone coordinates, next, we also need to normalize side chain atoms using the centroid

            # for the case of NaN value occurring in atom coordinates (currently try to arise errors)
            mask_nan = torch.isfinite(new_coords.sum(dim=(1, 2)))
            assert (~mask_nan).sum() == 0, 'Current pdb has invalid coordinates.' # if new_coords has invalid coordinates, original_coords will have them as well
            new_X_ca = new_coords[:, 1]

            # 3. start to generate features based on normalized coordinates
            # torch_cluster also includes radius graph
            new_edge_index = torch_cluster.knn_graph(new_X_ca, k=self.top_k) # knn_graph self loop default: False, the self-loop-like operation is realized in GVPConvLayer (formulas 4-5)
            # these features are generated based on corrupted coordinates
            # edge features (3)
            new_pos_embeddings = self._positional_embeddings(new_edge_index)
            new_E_vectors = new_X_ca[new_edge_index[0]] - new_X_ca[new_edge_index[1]]
            new_rbf = _rbf(new_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            # if the radius graph is needed
            if self.whether_spatial_graph:
                extra_edge_index = torch_cluster.radius_graph(new_X_ca, r=10.0)
                extra_pos_embeddings = self._positional_embeddings(extra_edge_index)
                extra_E_vectors = new_X_ca[extra_edge_index[0]] - new_X_ca[extra_edge_index[1]]
                extra_rbf = _rbf(extra_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            # node features
            # the below four type of features are the features specified in original GVP-GNN paper
            new_dihedrals = self._dihedrals(new_coords)
            new_orientations = self._orientations(new_X_ca) # torch.Size([126, 2, 3])
            new_sidechains = self._sidechains(new_coords)
            aatype_onehot = F.one_hot(torch.LongTensor([self.letter_to_num[i] for i in seq if i!=':']), num_classes = 20).to(self.device)

            # extra node feature
            # SASA, need to check the numerical range of SASA (currently choose not to normalize SASA)
            # original SASA file: ResidueID, ChainID, ASA_complex, ASA_single, dASA, Interface, currently SASA used in pretraining is asa_complex
            SASA = torch.as_tensor([float(sasa_dict[i]['asa_complex']) for i in residue_list], device=self.device, dtype=torch.float32) # every residue will be equipped with a SASA and an interface value
            # whether in complex interface or not
            interface = torch.as_tensor([int(sasa_dict[i]['interface']) for i in residue_list], device=self.device, dtype=torch.int32)
            # which chain current residue locates to
            chainid_seq = []
            for i in range(len(seq.split(':'))):
                chainid_seq.extend([i] * len(seq.split(':')[i]))
            chainid_seq = torch.as_tensor(chainid_seq, device=self.device, dtype=torch.int32)/i # dividing i is for normalization
            assert SASA.size() == interface.size() == chainid_seq.size(), 'SASA, interface, chainid_seq should have the same size in {}.'.format(name)

            # currently no surface aware features are provided here, because these features are more useful in protein-protein docking rather than here
            # https://github.com/octavian-ganea/equidock_public/blob/main/src/utils/protein_utils.py
            # current node feature overall
            # 1. aa type 2. SASA (objective) 3. three CA relevant features in GVP-GNN 4. whether on interface or not
            # 5. four side chain features (objective, vector) 6. side chain ids 7. normalized CA coordinates (objective: predicting relative distance between ground truth and perturbation coordinates, vector)
            assert aatype_onehot.size(0) == SASA.size(0) == new_dihedrals.size(0) == new_orientations.size(0) == new_sidechains.size(0) == interface.size(0)\
                   == chainid_seq.size(0) == new_X_ca.size(0), 'All generated features does not have the same length.'

            # 4. start to generate labels for each sub prediction tasks
            # SASA (mask the positions to be predicted)
            if self.remove_SASA_all != True:
                if self.SASA_mask == True:
                    SASA_label = SASA[mask].unsqueeze(-1)
                    SASA[mask] = 0
                else:
                    # in both cases, SASA_label is generated for other potential use
                    # while in this case, since SASA prediction is not treated as an auxiliary task, the corresponding SASA value in residues will be retained
                    SASA_label = SASA[mask].unsqueeze(-1)

            # extra AA prediction label
            if self.whether_AA_prediction:
                AA_prediction_label = aatype_onehot[mask].float()
                aatype_onehot[mask] = 0
            else:
                AA_prediction_label = aatype_onehot[mask].float()

            # merge all features
            if self.remove_SASA_all == True:
                node_s = torch.cat([aatype_onehot, new_dihedrals, chainid_seq.unsqueeze(-1)], dim=-1)
            else:
                node_s = torch.cat([aatype_onehot, SASA.unsqueeze(-1), new_dihedrals, interface.unsqueeze(-1), chainid_seq.unsqueeze(-1)], dim=-1)
            node_v = torch.cat([new_orientations, new_sidechains.unsqueeze(-2), new_X_ca.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([new_rbf, new_pos_embeddings], dim=-1)
            edge_v = _normalize(new_E_vectors).unsqueeze(-2)
            mask = torch.as_tensor(mask, device=self.device) # transform to tensor for letting PyG Data to stack masks of multiple proteins (as a complete tensor) correctly

            if self.whether_spatial_graph:
                extra_edge_s = torch.cat([extra_rbf, extra_pos_embeddings], dim=-1)
                extra_edge_v = _normalize(extra_E_vectors).unsqueeze(-2)

                if self.remove_SASA_all == True:
                    node_s, node_v, edge_s, edge_v, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, original_ca_label=original_ca_label, new_ca_label=new_ca_label,
                                                     AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask, extra_edge_index=extra_edge_index, extra_edge_s=extra_edge_s, extra_edge_v=extra_edge_v)
                else:
                    node_s, node_v, edge_s, edge_v, SASA_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, SASA_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, SASA_label=SASA_label, original_ca_label=original_ca_label, new_ca_label=new_ca_label,
                                                     AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask, extra_edge_index=extra_edge_index, extra_edge_s=extra_edge_s, extra_edge_v=extra_edge_v)

            else:
                if self.remove_SASA_all == True:
                    node_s, node_v, edge_s, edge_v, original_ca_label, new_ca_label, AA_prediction_label = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, original_ca_label, new_ca_label, AA_prediction_label))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, original_ca_label=original_ca_label, new_ca_label=new_ca_label,
                                                     AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask)
                else:
                    node_s, node_v, edge_s, edge_v, SASA_label, original_ca_label, new_ca_label, AA_prediction_label = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, SASA_label, original_ca_label, new_ca_label, AA_prediction_label))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, SASA_label=SASA_label, original_ca_label=original_ca_label, new_ca_label=new_ca_label,
                                                     AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask)

            return data

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


# after defining source data class, we need to define relevant pytorch Dataset class to produce required node and edge features
# here we need to consider how to add perturbation to coordinates etc. for gvp pre-training as well as surface aware features
class PretrainingGraphDataset_wosasa(data.Dataset):
    def __init__(self, data_list, noise_type: str, noise: float = 1.0, mask_prob: float = 0.15, only_CA = True, if_sidec_noise = True, num_positional_embeddings=16, top_k = 30, num_rbf=16, device='cpu',
                 sidec_chain_normalization = False, whether_AA_prediction = False, whether_spatial_graph = False, whether_sidec_prediction = False, whether_CG_feature = False, CG_encoding_type='sincos', CG_mask = False):
        super(PretrainingGraphDataset_wosasa, self).__init__()

        self.data_list = data_list # training/val list that stores each entry in the format of dict
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        # current 'seq' includes 'TER' symbol ':', thus we need to remove it when calculating node_counts
        # self.node_counts_ = [len(e['seq']) for e in data_list]
        self.node_counts = [len(''.join(e['seq'].split(':'))) for e in data_list] # used for passing residue node number to PyG DataLoader
        # print(list(zip([i['name'] for i in data_list], self.node_counts)))

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        self.atom_mass = {'C': 0.1621, 'N': 0.1891, 'O': 0.2160, 'S': 0.4328}
        # temp = 12.011 + 14.0067 + 15.9994 + 32.06
        # print(12.011/temp, 14.0067/temp, 15.9994/temp, 32.06/temp)

        self.whether_sidec_prediction = whether_sidec_prediction
        self.sidec_chain_normalization = sidec_chain_normalization  # for further normalization of side chain coordinates
        self.whether_AA_prediction = whether_AA_prediction # for adding an extra AA prediction task
        self.whether_spatial_graph = whether_spatial_graph
        self.whether_CG_feature = whether_CG_feature # indicating whether to use CG features
        self.CG_encoding_type = CG_encoding_type  # 'sin'/'onehot'/None

        # about generating noise for node coordinates
        self.noise_type = noise_type
        self.noise = noise # noise is used to control the magnitude of generated noise (similar to control the variance on the top of the standard normal distribution)
        self.mask_prob = mask_prob
        self.only_CA = only_CA # only perturb coordinates of CA in backbone (while the protein centroid is calculated based on C/CA/N/O in backbone), else disturb the whole backbone
        self.if_sidec_noise = if_sidec_noise # whether add noise to side chain atoms
        self.CG_mask = CG_mask # for determining whether to use CG as an auxiliary prediction task (given self.whether_CG_feature = True)
        self.counter = 0

        if self.noise_type == 'trunc_normal': # draw from normal distribution with truncated value
            self.noise_f = lambda num_mask: np.clip(
                np.random.randn(num_mask, 4, 3) * self.noise, a_min=-self.noise * 2.0, a_max=self.noise * 2.0) # in original version of uni-mol, the generated noise is [num_mask, 3] shape, currently is [num_mask, 4, 3] shape
        elif self.noise_type == 'normal':
            self.noise_f = lambda num_mask: np.random.randn(num_mask, 4, 3) * self.noise
        elif self.noise_type == 'uniform':
            self.noise_f = lambda num_mask: np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 4, 3))
        else:
            self.noise_f = lambda num_mask: 0.0

        if self.if_sidec_noise == True: # currently generate a noise for side chain atom set of each residue
            if self.noise_type == 'trunc_normal':
                self.sidec_noise_f = lambda num_mask: np.clip(
                    np.random.randn(num_mask, 3) * self.noise, a_min=-self.noise * 2.0, a_max=self.noise * 2.0)
            elif self.noise_type == 'normal':
                self.sidec_noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise
            elif self.noise_type == 'uniform':
                self.sidec_noise_f = lambda num_mask: np.random.uniform(
                    low=-self.noise, high=self.noise, size=(num_mask, 3))
            else:
                self.sidec_noise_f = lambda num_mask: 0.0

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        return self._featurize_as_graph(self.data_list[i], i)

    # *** need to check whether the generated mask and random coordinate perturbation will change in different epochs ***
    # *** need to ensure no operation will influence the original data stored in self.data_list (protein in _featurize_as_graph in originated from self.data_list[i]) ***
    # *** otherwise the data generated for the next batch will be changed, causing errors ***
    def _featurize_as_graph(self, protein, i):
        # self.counter += 1
        name = protein['name']
        # get complete entry of protein
        with torch.no_grad(): # close the gradiant trace
            seq = protein['seq']
            sasa_dict = protein['sasa_dict']
            # according to current logic, we need to corrupt coordinates at first, and generate features based on corrupted coordinates
            original_coords = protein['coords']
            # for solving the above problem, if we do not add deepcopy data_list[i]['sidechain_dict'] will change below
            # we need notice that the below way to process original_sidec will change the original self.data_list
            # original_sidec[key] = torch.as_tensor(np.array(original_sidec[key])[:, 1:].astype(np.float32), device=self.device, dtype=torch.float32)
            original_sidec = copy.deepcopy(protein['sidechain_dict'])

            # residue_list records the cr_token sequentially following the residue order of original pdb file
            residue_list = protein['res_idx_list']
            residue_num = len(residue_list) # protein['res_idx_list'] stores all residue identifier in current protein
            assert residue_num > 0, 'Residue number in current protein should be larger than 0: {}'.format(name)
            sidec_atom_list, atom_set = dict(), set()
            # record side chain atom type (following the residue_list order) and atom weight
            for res in residue_list:
                if res in original_sidec.keys():
                    # only retain the main atom name, like C/S/O/N
                    atoms = [i[0] for i in np.array(original_sidec[res])[:, 0]]
                    # atoms = []
                    # for i in np.array(original_sidec[res])[:, 0]:
                    #     atoms.append(i[0])
                    sidec_atom_list[res] = torch.as_tensor([self.atom_mass[j] for j in atoms], device=self.device, dtype=torch.float32).view(-1, 1)
                    for atom in atoms:
                        atom_set.add(atom)
            # print(atom_set) # {'C', 'S', 'O', 'N'} only C, S, O, N exist in side chain atoms of the pretraining set
            # record side chain atom coordinate tensor
            for key in original_sidec.keys():
                original_sidec[key] = torch.as_tensor(np.array(original_sidec[key])[:, 1:].astype(np.float32), device=self.device, dtype=torch.float32)

            # 1. start to generate residue mask
            # np.random.rand(): Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
            num_mask = int(self.mask_prob * residue_num + np.random.rand())
            mask_res = np.random.choice(residue_list, num_mask, replace=False).tolist() # replace=False: no duplicates, the order in mask_res is perturbed
            # print('mask_res:', len(mask_res), mask_res)
            mask = np.isin(residue_list, mask_res) # a numpy mask determining whether elements in A occur in B (np.isin(A, B)), in the order of residue_list
            # print('protein.keys():', protein.keys()) # dict_keys(['seq', 'coords', 'num_chains', 'name', 'sidechain_dict', 'sasa_dict', 'res_idx_list'])
            # start from here, we need to generate the corrupted residue coordinates (the masked residue positions are recorded by 'mask')
            # currently follow the scheme that, generating all node and edge features after corrupting coordinates
            num_mask = mask.astype(np.int32).sum() # boolean to int, mask value = True/1 represents this residue will be corrupted

            # 2. start to process coordinates, all features generated will be based on new coordinates
            new_coords = np.copy(np.array(original_coords)) # (126, 4, 3)
            if self.only_CA == True:
                temp_noise = self.noise_f(num_mask)[:, 1]
                new_coords[mask, 1] += temp_noise
                # print('current noise:', temp_noise[0]) # print the noise generated for the first residue of current protein
            else:
                temp_noise = self.noise_f(num_mask)
                new_coords[mask, :] += temp_noise

            # all coordinates in new_coords should be retained, as we need to the whole new_coords to generate residue graph and features
            # and then retrieve the corrupted residues through mask
            original_coords = torch.as_tensor(original_coords, device=self.device, dtype=torch.float32)
            new_coords = torch.as_tensor(new_coords, device=self.device, dtype=torch.float32)

            # because for the CA denoising task, what gvp predicts is the relative position between absolute original coordinates and absolute perturbed coordinates
            # thus we need to get absolute CA coordinates here before coordinate normalization as the labels
            # the below generated original_X_ca and new_X_ca are based on normalized coordinates for calculating other labels/features, like side chain geometric information
            original_ca_label = original_coords[:, 1][mask] # atom order here: N, CA, C, O, based on absolute coordinates
            new_ca_label = new_coords[:, 1][mask]

            # currently, the centroid of current protein is based on the coordinates of all backbone atoms (N, CA, C, O) rather than just based on CA
            original_centroid = self.centroid(original_coords)
            new_centroid = self.centroid(new_coords)
            original_coords = original_coords - original_centroid
            new_coords = new_coords - new_centroid # this is the normalization of new backbone coordinates, next, we also need to normalize side chain atoms using the centroid

            # for the case of NaN value occurring in atom coordinates (currently try to arise errors)
            mask_nan = torch.isfinite(new_coords.sum(dim=(1, 2)))
            assert (~mask_nan).sum() == 0, 'Current pdb has invalid coordinates.' # if new_coords has invalid coordinates, original_coords will have them as well
            original_X_ca = original_coords[:, 1] # based on normalized coordinates, to generate other relevant features
            new_X_ca = new_coords[:, 1]

            # need to consider the correspondence between corrupted side chain coordinates and the ground truth coordinates
            # current solution: after deepcopy, for each residue, its side chain atom order will not change
            new_sidec = copy.deepcopy(original_sidec)
            for key in original_sidec.keys():
                original_sidec[key] = original_sidec[key] - original_centroid
            for key in new_sidec.keys(): # side chain coordinate normalization
                new_sidec[key] = new_sidec[key] - new_centroid

            # *** actually, if we add noise to residue coordinates and generate side chain atom feature based on the relative position between residue and side chain coordinates ***
            # *** in this case, maybe it is also feasible to just add noise to the residue rather than adding noise to side chain atom as well, because the relative position will change simultaneously when residue noise is added ***
            # *** furthermore, the denoising process includes residue coordinate denoising and the side chain relative position denoising (sub-tasks) ***
            # *** if set if_sidec_noise to true, the relevant residue node embedding will change, in other words, even though the corresponding loss is not calculated, it will also influence the prediction of other tasks ***
            if self.if_sidec_noise:
                sorted_mask_res = sorted(mask_res)
                assert num_mask == len(sorted_mask_res), 'relevant error pdb name: {}, num_mask: {}, len(sorted_mask_res): {}'.format(name, num_mask, len(sorted_mask_res))
                sidec_noise_f = torch.as_tensor(self.sidec_noise_f(num_mask), device=self.device, dtype=torch.float32)
                for i in range(num_mask):
                    # need to consider one case that the side chain atom coordinates to be perturbed do not exist in current pdb
                    if sorted_mask_res[i] in new_sidec.keys():
                        # previous logic: add the same noise to every atom coordinate in the side chain
                        # new_sidec[sorted_mask_res[i]] = new_sidec[sorted_mask_res[i]] + sidec_noise_f[i]
                        # new logic: randomly select one side chain atom and add noise that so that the centroid of side chain atom set could be broken and be denoising later
                        random_index = random.sample(range(new_sidec[sorted_mask_res[i]].size(0)), 1)
                        new_sidec[sorted_mask_res[i]][random_index] = new_sidec[sorted_mask_res[i]][random_index] + sidec_noise_f[i]

            # 3. start to generate features based on normalized coordinates
            # torch_cluster also includes radius graph
            new_edge_index = torch_cluster.knn_graph(new_X_ca, k=self.top_k) # knn_graph self loop default: False, the self-loop-like operation is realized in GVPConvLayer (formulas 4-5)
            # these features are generated based on corrupted coordinates
            # edge features (3)
            new_pos_embeddings = self._positional_embeddings(new_edge_index)
            new_E_vectors = new_X_ca[new_edge_index[0]] - new_X_ca[new_edge_index[1]]
            new_rbf = _rbf(new_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            # if the radius graph is needed
            if self.whether_spatial_graph:
                extra_edge_index = torch_cluster.radius_graph(new_X_ca, r=10.0)
                extra_pos_embeddings = self._positional_embeddings(extra_edge_index)
                extra_E_vectors = new_X_ca[extra_edge_index[0]] - new_X_ca[extra_edge_index[1]]
                extra_rbf = _rbf(extra_E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            # node features
            # the below four type of features are the features specified in original GVP-GNN paper
            new_dihedrals = self._dihedrals(new_coords)
            new_orientations = self._orientations(new_X_ca) # torch.Size([126, 2, 3])
            new_sidechains = self._sidechains(new_coords)
            aatype_onehot = F.one_hot(torch.LongTensor([self.letter_to_num[i] for i in seq if i!=':']), num_classes = 20).to(self.device)

            # extra node feature
            # which chain current residue locates to
            chainid_seq = []
            for i in range(len(seq.split(':'))):
                chainid_seq.extend([i] * len(seq.split(':')[i]))
            chainid_seq = torch.as_tensor(chainid_seq, device=self.device, dtype=torch.int32)/i # dividing i is for normalization

            # residue_list records the cr_token sequentially following the residue order of original pdb file
            # currently the below four four chain coordinate information has not been further normalized and the vacant places are filled with zero
            if self.sidec_chain_normalization:
                # current centroid of protein backbone complex is calculated based on original_coords (C+CA+N+O), thus the further normalization is also based on this
                # currently original_coords has been through centroid normalization
                length = torch.sqrt(torch.sum((new_coords.view(-1, 3) ** 2), -1))
                length = length[torch.argmax(length)]
                sidec_seq_max = (torch.concat([torch.max(new_sidec[i], dim=0, keepdim=True)[0] if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                    .type(torch.float32).to(self.device)
                sidec_seq_centroid = (torch.concat([torch.mean(new_sidec[i], dim=0, keepdim=True) if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                    .type(torch.float32).to(self.device)
                sidec_seq_mass = (torch.concat([torch.mean(new_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                    .type(torch.float32).to(self.device)
                sidec_CA_relations = self._CA_to_sidechains(new_X_ca, new_sidec, residue_list)  # relative value, not influenced by the further normlization
                new_X_ca = new_X_ca / length
            else:
                sidec_seq_max = torch.concat([torch.max(new_sidec[i], dim=0, keepdim=True)[0] if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                    .type(torch.float32).to(self.device)
                sidec_seq_centroid = torch.concat([torch.mean(new_sidec[i], dim=0, keepdim=True) if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                    .type(torch.float32).to(self.device)
                sidec_seq_mass = torch.concat([torch.mean(new_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in new_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                    .type(torch.float32).to(self.device) # also based on new_centroid
                # print(new_sidec['A_1'].size(), sidec_atom_list['A_1'].size(), sidec_seq_mass.size())
                sidec_CA_relations = self._CA_to_sidechains(new_X_ca, new_sidec, residue_list)

            # currently no surface aware features are provided here, because these features are more useful in protein-protein docking rather than here
            # https://github.com/octavian-ganea/equidock_public/blob/main/src/utils/protein_utils.py
            # current node feature overall
            # 1. aa type 2. SASA (objective) 3. three CA relevant features in GVP-GNN 4. whether on interface or not
            # 5. four side chain features (objective, vector) 6. side chain ids 7. normalized CA coordinates (objective: predicting relative distance between ground truth and perturbation coordinates, vector)
            assert aatype_onehot.size(0) == new_dihedrals.size(0) == new_orientations.size(0) == new_sidechains.size(0) == sidec_seq_max.size(0) == sidec_seq_centroid.size(0) == sidec_seq_mass.size(0)\
                   == sidec_CA_relations.size(0) == chainid_seq.size(0) == new_X_ca.size(0), 'All generated features does not have the same length.'

            # 4. start to generate labels for each sub prediction tasks
            # we need calculate the original corresponding features
            if self.whether_sidec_prediction == True:
                if self.sidec_chain_normalization:
                    length = torch.sqrt(torch.sum((original_coords.view(-1, 3) ** 2), -1))
                    length = length[torch.argmax(length)]
                    sidec_seq_max_label = (torch.concat([torch.max(original_sidec[i], dim=0, keepdim=True)[0] if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_seq_centroid_label = (torch.concat([torch.mean(original_sidec[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_seq_mass_label = (torch.concat([torch.mean(original_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) / length) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_CA_relations_label = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list)[mask]  # relative value, not influenced by the further normlization
                else:
                    sidec_seq_max_label = torch.concat([torch.max(original_sidec[i], dim=0, keepdim=True)[0] if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_seq_centroid_label = torch.concat([torch.mean(original_sidec[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                        .type(torch.float32).to(self.device)[mask]
                    sidec_seq_mass_label = torch.concat([torch.mean(original_sidec[i] * sidec_atom_list[i], dim=0, keepdim=True) if i in original_sidec.keys() else torch.zeros(1, 3) for i in residue_list]) \
                        .type(torch.float32).to(self.device)[mask] # also based on new_centroid
                    # currently _CA_to_sidechains is based on normalized CA coordinates instead of the absolute ones
                    sidec_CA_relations_label = self._CA_to_sidechains(original_X_ca, original_sidec, residue_list)[mask]
                sidec_label = torch.cat([sidec_seq_max_label.unsqueeze(-2), sidec_seq_centroid_label.unsqueeze(-2), sidec_seq_mass_label.unsqueeze(-2), sidec_CA_relations_label], dim=-2)
            else:
                sidec_label = torch.zeros(chainid_seq.size(0), 4).to(self.device)

            # extra AA prediction label
            if self.whether_AA_prediction:
                AA_prediction_label = aatype_onehot[mask].float()
                aatype_onehot[mask] = 0
            else:
                AA_prediction_label = aatype_onehot[mask].float()

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
                    cg_feature = torch.LongTensor(cg_feature).to(self.device)
                    cg_feature = torch.cat([F.one_hot(cg_feature[:, i], num_classes=pos_feats_num[i]) for i in range(pos_feats_num.size(0))], dim=-1).float()
                else:
                    cg_feature = torch.as_tensor(cg_feature, device=self.device, dtype=torch.float32) / pos_feats_num
                assert aatype_onehot.size(0) == cg_feature.size(0)  # check for the size of cg_feature

                if self.CG_mask == True:
                    CG_label = cg_feature[mask]
                    cg_feature[mask] = 0
                # merge all information and return it
                node_s = torch.cat([aatype_onehot, new_dihedrals, chainid_seq.unsqueeze(-1), cg_feature], dim=-1)
            else:
                # print(aatype_onehot.size(), SASA.size(), new_dihedrals.size(), new_orientations.size(), new_sidechains.size(), interface.size(),
                #       sidec_seq_max.size(), sidec_seq_centroid.size(), sidec_seq_mass.size(), sidec_CA_relations.size(), chainid_seq.size(), new_X_ca.size())
                # torch.Size([126, 20]) torch.Size([126]) torch.Size([126, 6]) torch.Size([126, 2, 3]) torch.Size([126, 3]) torch.Size([126])
                # torch.Size([126, 3]) torch.Size([126, 3]) torch.Size([126, 3]) torch.Size([126, 2, 3]) torch.Size([126]) torch.Size([126, 3])
                node_s = torch.cat([aatype_onehot, new_dihedrals, chainid_seq.unsqueeze(-1)], dim=-1)

            node_v = torch.cat([new_orientations, new_sidechains.unsqueeze(-2), sidec_seq_max.unsqueeze(-2), sidec_seq_centroid.unsqueeze(-2), sidec_seq_mass.unsqueeze(-2), sidec_CA_relations, new_X_ca.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([new_rbf, new_pos_embeddings], dim=-1)
            edge_v = _normalize(new_E_vectors).unsqueeze(-2)
            mask = torch.as_tensor(mask, device=self.device) # transform to tensor for letting PyG Data to stack masks of multiple proteins (as a complete tensor) correctly

            if self.whether_spatial_graph:
                extra_edge_s = torch.cat([extra_rbf, extra_pos_embeddings], dim=-1)
                extra_edge_v = _normalize(extra_E_vectors).unsqueeze(-2)
                if self.whether_CG_feature and self.CG_mask:
                    node_s, node_v, edge_s, edge_v, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v, CG_label = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v, CG_label))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, sidec_label=sidec_label, original_ca_label=original_ca_label,
                                                     new_ca_label=new_ca_label, AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask,
                                                     extra_edge_index=extra_edge_index, extra_edge_s=extra_edge_s, extra_edge_v=extra_edge_v, CG_label=CG_label)
                else:
                    node_s, node_v, edge_s, edge_v, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, extra_edge_s, extra_edge_v))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, sidec_label=sidec_label, original_ca_label=original_ca_label,
                                                     new_ca_label=new_ca_label, AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask,
                                                     extra_edge_index=extra_edge_index, extra_edge_s=extra_edge_s, extra_edge_v=extra_edge_v)
            else:
                if self.whether_CG_feature and self.CG_mask:
                    node_s, node_v, edge_s, edge_v, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, CG_label = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, sidec_label, original_ca_label, new_ca_label, AA_prediction_label, CG_label))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, sidec_label=sidec_label, original_ca_label=original_ca_label,
                                                     new_ca_label=new_ca_label, AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask, CG_label=CG_label)
                else:
                    node_s, node_v, edge_s, edge_v, sidec_label, original_ca_label, new_ca_label, AA_prediction_label = map(
                        torch.nan_to_num, (node_s, node_v, edge_s, edge_v, sidec_label, original_ca_label, new_ca_label, AA_prediction_label))
                    data = torch_geometric.data.Data(node_s=node_s, node_v=node_v, edge_s=edge_s, edge_v=edge_v, sidec_label=sidec_label, original_ca_label=original_ca_label,
                                                     new_ca_label=new_ca_label, AA_prediction_label=AA_prediction_label, edge_index=new_edge_index, mask=mask)
                    # Data(edge_index=[2, 3780], name='1a0a_1', node_s=[126, 29], node_v=[126, 9, 3], edge_s=[3780, 32], edge_v=[3780, 1, 3], SASA_label=[19], sidec_label=[19, 5, 3], X_ca_label=[19, 3], mask=[126])

            return data

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


if __name__ == '__main__':
    # test example
    temp_root = './data/simulation_dataset/'
    dataset = PretrainingDataset(temp_root + 'pretraining_chain_set.jsonl', temp_root + 'pretraining_data_split.json')
    print('dataset.train:', len(dataset.train)) # dataset.train/val is a protein entry list for training/validation
    print('dataset.val:', len(dataset.val))

    # *read the list stored in PretrainingDataset as the input of pytorch Dataset*
    # part of the input of PretrainingGraphDataset:
    # data_list; noise_type: str; noise: float = 1.0; mask_prob: float = 0.15; only_CA = True; if_sidec_noise = True; SASA_mask = True; num_positional_embeddings=16; top_k = 30; num_rbf=16; device='cpu';
    train_set = PretrainingGraphDataset(dataset.train, 'uniform', 1.0, 0.15, True, True, True)
    # val_set = PretrainingGraphDataset(dataset.val, 'uniform', 1.0, 0.15, True, False, True)

    for i in dataset.val:
        if i['name'] == '5u4k':
            test_pdb = i
            break

    train_feats_example = train_set._featurize_as_graph(test_pdb, 0)
    print('train_feats_example:', train_feats_example)
