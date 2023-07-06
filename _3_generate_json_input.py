# step3:
# after completing side chain atoms for each dataset
# next we need to call relevant function to generate input for pytorch Dataset, to get gvp input for each sample (_3_generate_residuefeats)
import os
import json
import pandas as pd
import numpy as np


def main_chain_processing(coords_folder_path, sasa_folder_path, pdb):
    # print('pdb name:', pdb)
    # need the information for both backbones and side chains
    num_chains = 0
    sidechain_dict = dict()
    sasa_dict = dict()  # storing sasa related information based on cr_token format
    chain_set = set()
    name = pdb[:-4]

    N_list = []
    CA_list = []
    C_list = []
    O_list = []
    AA_list = []  # residue type list
    res_idx_list = []  # effective residue sequence identifiers (based on cr_token, excluding non-natural AAs), for retrieving side chain atoms sequentially
    complete_res_idx_list = [] # complete residue sequence identifiers

    residues = ['ARG', 'MET', 'VAL', 'ASN', 'PRO', 'THR', 'PHE', 'ASP', 'ILE',
                'ALA', 'GLY', 'GLU', 'LEU', 'SER', 'LYS', 'TYR', 'CYS', 'HIS', 'GLN', 'TRP']
    res_code = ['R', 'M', 'V', 'N', 'P', 'T', 'F', 'D', 'I',
                'A', 'G', 'E', 'L', 'S', 'K', 'Y', 'C', 'H', 'Q', 'W']
    res2code = {x: idxx for x, idxx in zip(residues, res_code)}
    backbone_atom = ['C', 'CA', 'N', 'O']

    # use pdb name to read the pdb file and corresponding sasa file simultaneously
    with open(coords_folder_path + pdb) as f:
        lines = f.readlines()
    with open(sasa_folder_path + pdb + '.sasa') as f:
        sasa_lines = f.readlines()

    def get_coords(line):
        if len(line[30:38].strip()) != 0:
            x = float(line[30:38].strip())
        else:
            x = float('nan')  # nan in math format
        if len(line[38:46].strip()) != 0:
            y = float(line[38:46].strip())
        else:
            y = float('nan')
        if len(line[46:54].strip()) != 0:
            z = float(line[46:54].strip())
        else:
            z = float('nan')
        return x, y, z

    # before formally generating json file, we need to ensure that cr_token is unique in a protein
    # in some extreme cases like 5u4k, the same cr_token is repeated multiple times (586A-672A, 519B-551B), causing errors on generating residue coordinate denoising masks
    # this is because some proteins are observed by NMR, it can capture the dynamics of proteins
    # creating multiple conformations of one protein into one pdb file (sometimes separated by 'model' tag in raw pdb file), each repeat using the same cr_token label
    # thus, we need to consider such proteins for providing effective data to pytorch DataLoader
    # in the last step (using foldx to complete side chain atoms), giving chain a new id is only conditioned on lines[0][21].isdigit(), i.e., there are some digital chain ids in a protein
    # thus, such NMR generated pdbs are not detected by the last step

    # need to consider residue serial number like 1A/1B (line[22:28].strip())
    # counter1 = 0
    residue_flag, residue_flag_ = -1, -1
    for line in lines:
        # here we do not need to consider to remove H atoms, HETATM atoms and AASP/BASP cases, as they are removed in the previous two steps
        # about retrieving residue types, for current plan, we only consider the main 20 AA types
        # for non-natural residues, they will be ignored in coordinate record and AA_list record (AA_list is for potential alphafold generation, and it cannot directly handle non-natural residues, thus deleting them)
        if line[0:4] == 'ATOM':
            resname = line[16:21].strip()  # residue name
            chainid = line[21] # chain id
            res_idx = line[22:28].strip()  # current residue idx, can solve the case of "H_100A", "H_100B", and "H_100C"
            cr_token = '{}_{}'.format(chainid, res_idx)
            if residue_flag_ != cr_token: # record the complete AA sequence
                complete_res_idx_list.append(cr_token)
                residue_flag_ = cr_token
            if residue_flag != cr_token: # enter a new residue, record the effective AA sequence
                if resname[-3:] in residues: # in the case of 4-digit residue name
                    res_idx_list.append(cr_token)
                    AA_list.append(res2code[resname])
                    residue_flag = cr_token
                else:
                    print('non-natural residue line in {}:'.format(name), line) # allow to print
                    continue # skip current non-natural line for detection

        elif line[0:3] == 'TER':
            AA_list.append(':')
            # current residue_flag is the final residue idx before 'TER'
            # temp = residue_flag
            # temp = [i for i in temp if not i.isalpha()]
            # counter1 += int(''.join(temp))
            # counter1 += 1
            num_chains += 1

        if line[0:4] == 'ATOM':
            atomname = line[12:16].strip()
            chainid = line[21]
            res_idx = line[22:28].strip()  # get residue idx in current line (only atom line consider res_idx)
            cr_token = '{}_{}'.format(chainid, res_idx)
            # about retrieving backbone coordinates
            if atomname == 'N':
                N_x, N_y, N_z = get_coords(line)  # need the complete line as input
                N_list.append([N_x, N_y, N_z])
            elif atomname == 'CA':
                CA_x, CA_y, CA_z = get_coords(line)
                CA_list.append([CA_x, CA_y, CA_z])
            elif atomname == 'C':
                C_x, C_y, C_z = get_coords(line)
                C_list.append([C_x, C_y, C_z])
            elif atomname == 'O':
                O_x, O_y, O_z = get_coords(line)
                O_list.append([O_x, O_y, O_z])

            # about retrieving side chain coordinates (a group of coordinates for a residue and excluding H)
            # need to generate a list based on the ordered cr_token for retrieving them sequentially
            # elif not atomname.endswith(('CA', 'HA', 'N', 'C', 'O', 'HN', 'H')) and not atomname.startswith(('H')): # previous judgement condition
            # the atom orders from foldx and alphafold2 in pdb could be different
            # updated judgement condition, recording atoms except for the listed element type (for solving af2 generated strucutres):
            elif (atomname not in set(backbone_atom)) and (not atomname.startswith(('H'))):
                sc_x, sc_y, sc_z = get_coords(line)
                if cr_token not in sidechain_dict.keys():  # every cr_token represents an individual residue in current protein
                    sidechain_dict[cr_token] = []
                    sidechain_dict[cr_token].append([atomname, sc_x, sc_y, sc_z])
                else:
                    sidechain_dict[cr_token].append([atomname, sc_x, sc_y, sc_z])

    # 1. after being processed by foldx, the last 'TER' of pdb will be removed
    # 2. for pretraining set, before sending pdbs to foldx for side chain completion, we need to use _1_run_cleaning_pdb.py to only retain 'ATOM' and 'TER' lines
    # 3. in contrast, the original widetype pdb files before being processed by _1_run_cleaning_pdb.py and foldx
    # and the (original) alphafold generated pdb files contain the last 'TER' (and 'END')
    # 4. to keep the consistency between pretraining and downstream sets, currently, for pretraining set, using _1_run_cleaning_pdb.py and foldx to preprocess
    # for downstream set, using foldx and _1_run_cleaning_pdb.py to preprocess, the aim is to only retain 'ATOM' and 'TER' lines in each pdb
    # 5. after this step, _3_generate_json_input.py can be used to generate json file for pytorch dataloader
    # 6. to further keep the consistency, we will also preprocess alphafold generated pdb files using
    # 7. besides, the chainid_seq feature generation logic in _featurize_as_graph requires no ':' occurring in the last line of AA_list
    if line[0:3] != 'TER': # until this step, only 'ATOM' and 'TER' will be retained in the pdb (foldx format, the last 'TER' will not be retained)
        num_chains += 1

    # sort the atom set in each side chain (for the consistency between WT and MT)
    for key in sidechain_dict.keys():
        sidechain_dict[key] = sorted(sidechain_dict[key])

    # print(counter1, len(AA_list))  # could be different, because in a chain the residue serial number may not start from 1
    # assert counter1 == len(AA_list), 'The AA sequence length does not correspond to current pdb file {}.'.format(name)
    assert len(N_list) == len(CA_list) == len(C_list) == len(O_list), 'The length of coordinate lists is different for backbone atoms in {}'.format(pdb)
    seq = ''.join(AA_list)
    coords = {'N': N_list, 'CA': CA_list, 'C': C_list, 'O': O_list}
    # the entry number in sidechain_dict and res_idx_list could be different
    # because some residues could not have side chain atoms while res_idx_list just records all resiude identifers sequentially
    # print('len(sidechain_dict), len(res_idx_list):', len(sidechain_dict), len(res_idx_list))

    # for considering the repeated cr_token protein mentioned above (current strategy for these proteins is to ignore them, not letting them into pretraining set)
    # in ideal case, if these proteins can be detected successfully, it can be detached into multiple pdbs with which independent SASA files can be generated, for enlarging pretraining sample number
    if len(sasa_lines)-1 != len(res_idx_list): # *** current logic for SASA generation is based on effective natural AAs (also ignoring non-natural AAs) ***
        # in most cases, len(res_idx_list) % (len(sasa_lines)-1) = 0 represents the case about NMR repeated proteins, otherwise represents other mismatch between pdb and corresponding SASA values (e.g., 3_mgn_1.pdb)
        print('repeated cr_token protein name:', pdb, 'sasa_lines:', len(sasa_lines)-1, 'res_idx_list:', len(res_idx_list), len(res_idx_list) % (len(sasa_lines)-1))

        with open('./data/data_information/{}_repeated_cr_token_protein.txt'.format(coords_folder_path.split('/')[-2]), 'a') as f:
            f.write(
                'repeated cr_token protein name: {}'.format(pdb) + 'sasa_lines: {}'.format(len(sasa_lines)-1) + 'res_idx_list: {}, {}'.format(len(res_idx_list), len(res_idx_list) % (len(sasa_lines)-1)) + '\n')
        return 0

    # current cr_token: the last cr_token in the pdb file
    # this check will not be used, as cr_token_check1 and cr_token could be different, because current SASA is generated based on effective AAs, if the last cr_token in WT coordinates is a non-natural AA
    # it will be removed in corresponding SASA files, leading to the unconsistency (e.g., 1U7F.pdb)
    # cr_token_check1 = '{}_{}'.format(sasa_lines[-1].split()[1], sasa_lines[-1].split()[0])
    # assert cr_token_check1 == cr_token, 'cr tokens: {}, {} in {}'.format(cr_token, cr_token_check1, name)

    # after generating information about coordinates, start to generate information about SASA and complex interface
    counter2 = 0
    # adding res_idx_list into the loop to check the inconsistency between coordinate data and sasa data
    # e.g., 1B-1, 1A-1, 1-1 (residue serial number) coordinate data vs sasa data
    assert len(sasa_lines)-1 == len(res_idx_list), 'cr token number should be the same in the SASA file and corresponding effective coordinate file for {}'.format(name)
    # *** current logic for SASA generation is based on effective natural AAs (also ignoring non-natural AAs), thus we use res_idx_list rather than complete_res_idx_list to run loop check ***
    for sasaline, check in zip(sasa_lines, ['start'] + res_idx_list):
        # *** this check is only for checking the residue id consistency, if the chain id is changed while residue id is not changed, the check for this case will pass (but may cause error in model finetuning) ***
        counter2 += 1
        if counter2 == 1:  # removing the title line
            continue
        # residue id, chain id, asa_complex, asa_single, dasa, interface
        # if a protein has at least two chains, DASA can be calculated and then the interface can be correctly calculated

        res_idx = sasaline.split()[0] # get the residue serial number provided in SASA file (res_idx)
        res_check = check.split('_')[-1] # get the residue serial number provided in coordinate file (check_), check: L_1B, check_: 1B, res_idx: 1
        # remove alpha letters in res_idx and check_
        # in some extreme cases like 2f9z.pdb, it has cr_token D_-1, in this case '-' should be retained, what needs to be removed is just alpha
        s = ''.join([i for i in res_idx if not i.isalpha()])
        v = ''.join([i for i in res_check if not i.isalpha()])
        # *** we require the exact correspondence between AA serial numbers in the pdbs and their SASA files ***
        # *** but we do not require that the AA serial numbers must be consecutive ***
        assert s == v, 'current residue serial number in SASA: {}, in coords: {} (in {})'.format(res_idx, res_check, pdb)
        if res_check != res_idx:
            res_idx = res_check # subject to the serial number in coordinate file

        # only SASA that is related to natural AAs will be retained (based on cr_tokens in res_idx_list)
        # res_idx_list is obtained based on the natural AA sequence of original pdb coordinates above
        if check in res_idx_list:
            chainid = sasaline.split()[1]
            asa_complex = sasaline.split()[2]
            asa_single = sasaline.split()[3]
            dasa = sasaline.split()[4]
            interface = sasaline.split()[5]
            # each residue will occur one time
            sasa_dict['{}_{}'.format(chainid, res_idx)] = {'asa_complex': asa_complex, 'asa_single': asa_single, 'dasa': dasa, 'interface': interface}
        else:
            print('current cr_token is in non-natural amino acid:', name, check)

    # AA sequence, backbone coordinates, chain number, protein, side chain atoms, residue sasa information, residue sequence identifiers
    # return seq, coords, num_chains, name, sidechain_dict, sasa_dict, res_idx_list
    return {'seq': seq, 'coords': coords, 'num_chains': num_chains, 'name': name,
            'sidechain_dict': sidechain_dict, 'sasa_dict': sasa_dict, 'res_idx_list': res_idx_list, 'len_complete_aalist': len(complete_res_idx_list), 'len_effective_aalist': len(res_idx_list)}


def json_generator(task, root_path, dataset_name1, coords_folder_path1, sasa_folder_path1, dataset_name2=None, coords_folder_path2=None, sasa_folder_path2=None, save_num=None):
    # preliminary check between coordinates and sasa files
    pdbs1 = sorted(os.listdir(root_path + coords_folder_path1))
    sasa1 = sorted(os.listdir(root_path + sasa_folder_path1))
    print('file number in pdbs1 and sasa1:', len(pdbs1), len(sasa1))
    assert len(pdbs1) == len(sasa1), 'pdb coordinate file number: {}, corresponding SASA file number: {}'.format(len(pdbs1), len(sasa1))
    for i, j in zip(pdbs1, sasa1):
        if i.split('.')[0] != j.split('.')[0]:
            print('mismatching in wild type files:', i, j)
    # for mutation files for downstream tasks
    if dataset_name2 and coords_folder_path2 and sasa_folder_path2:
        pdbs2 = sorted(os.listdir(root_path + coords_folder_path2))
        sasa2 = sorted(os.listdir(root_path + sasa_folder_path2))
        assert len(pdbs2) == len(sasa2), 'pdb coordinate file number: {}, corresponding SASA file number: {}'.format(len(pdbs2), len(sasa2))
        for i, j in zip(pdbs2, sasa2):
            if i.split('.')[0] != j.split('.')[0]:
                print('mismatching in mutation files:', i, j)

    # for json generation for pretraining task:
    if task == 'pretraining':
        print('generating json file for pretraining')
        # data generation loop (here we need to screen overlarge protein again for reducing computational cost)
        if os.path.exists('./data/pretraining_chain_set.jsonl'):
            print('pretraining_chain_set.jsonl has been generated')
        else:
            with open('./data/pretraining_chain_set.jsonl', 'w') as f1:
                counter1, counter3 = 0, 0
                for pdb1 in pdbs1: # the pdbs order is fixed above
                    counter1 += 1
                    if counter1 % 2000 == 0:
                        print('counter:', counter1)
                    if save_num:
                        if counter1 % save_num == 0:
                            break
                    with open(root_path + coords_folder_path1 + pdb1) as f2: # read all pdbs in the specified folder
                        lines = f2.readlines()

                    counter2 = 0
                    for line in lines:
                        if line[0:4] == 'ATOM' and line[12:16].strip() == 'CA':
                            counter2 += 1
                    if counter2 >= 3000:
                        with open('./data/data_information/{}_overlarge_protein.txt'.format(dataset_name1), 'a') as f3:
                            f3.write(pdb1 + '\n')
                        print('overlarge protein: {}'.format(pdb1))
                        continue
                    else: # write to the json file
                        # use pdb name to read the pdb file and corresponding sasa file simultaneously
                        data_dict = main_chain_processing(root_path + coords_folder_path1, root_path + sasa_folder_path1, pdb1)
                        # for the case that repeated cr_token protein occurs
                        # the relevant information will be printed from main_chain_processing function
                        if data_dict == 0:
                            counter3 += 1
                        else:
                            f1.write(json.dumps(data_dict) + '\n') # write the json file for current pdb
                print('repeated cr_token protein number:', counter3)

    # for json generation for downstream tasks:
    # here we consider to add the labels of downstream tasks (into json), after the json generation, the json file will be sent to pytorch Dataset being split by data-spliting file directly
    elif task == 'finetuning':
        print('generating json file for finetuning')
        # every mutation complex will correspond to an independent widetype complex (specified by counters)
        # here the pdb and sasa file numbers of widetype and mutation should be the same, but the residue number in each file could be different between widetype and mutation (and may need to adjust)
        assert len(pdbs1) == len(pdbs2) == len(sasa1) == len(sasa2), 'file numbers in each folder: {}, {}, {}, {}'.format(len(pdbs1), len(pdbs2), len(sasa1), len(sasa2))

        # load the original mutation file (and check whether dataset_name1 == dataset_name2)
        if dataset_name1 == dataset_name2 == 'M1101':
            mutation_set = pd.read_csv(root_path + dataset_name1 + '.csv', encoding='latin-1')
        elif dataset_name1 == dataset_name2 and dataset_name1 != 'M1101' :
            mutation_set = pd.read_csv(root_path + dataset_name1 + '.csv')

        if dataset_name1 == 'S645':
            mutation_set = mutation_set[['#PDB', 'Partners(A_B)', 'Mutation', 'ddG(kcal/mol)']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
        elif dataset_name1 == 'M1101':
            mutation_set = mutation_set[['#PDB', 'Partners(A_B)', 'Mutation', 'ddG(kcal/mol)']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
        elif dataset_name1 == 'S1131':
            mutation_set = mutation_set[['protein', 'Partners(A_B)', 'mutation', 'DDG']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
        elif dataset_name1 == 'M1707':
            mutation_set = mutation_set[['PDB id', 'Partner1', 'Mutation(s)_cleaned', 'DDGexp']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
            # for M1707, for consistency, change its mutation information to 'D:A488G' mode
            # The mutation information includes WT residue, chain, residue index and mutant residue. such as “TI38F”
            # which stands for mutating the 38th acid amino at the I chain (i.e., phenylalanine) to threonine.
            mutation_ = []
            for row in mutation_set['mutation']:
                temp = ','.join([i[1] + ':' + i[0] + i[2:-1] + i[-1] for i in row.split(',')]) # change to 'D:A488G' mode
                mutation_.append(temp)
            mutation_set['mutation'] = mutation_
            # print(mutation_set)
        elif dataset_name1 == 'S4169':
            mutation_set = mutation_set[['protein', 'Partners(A_B)', 'mutation', 'DDG']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']

        mutation_set_np = np.array(mutation_set) # pdb, partner, mutation, ddg

        # load the interface and mutation file
        with open(root_path + dataset_name1 + '_interface_mutation_dict.json') as f:
            interface_mutation = json.load(f)

        if os.path.exists(f'./data/{dataset_name1}_chain_set.jsonl'):
            print(f'{dataset_name1}_chain_set.jsonl has been generated in ./data/')
        else:
            with open(f'./data/{dataset_name1}_chain_set.jsonl', 'w') as f1: # for jsonl, every entry in it is a complete json file for a mutation sample

                counter1, counter2, counter3 = 0, 0, 0
                for mutation in mutation_set_np[:, [0, 1, 2]]:
                    counter1 += 1
                    if counter1 % 2000 == 0:
                        print('counter:', counter1)
                    if save_num:
                        if counter1 % (save_num + 1) == 0:
                            break
                    pdb_name = mutation[0] + '__' + str(counter1) + '.pdb' # ['pdb', 'partner', 'mutation', 'ddg']

                    # some pdbs may be ignored during previous pre-processing steps, thus the generated json file only contains pdbs in the specified folder
                    if pdb_name in pdbs1 and pdb_name in pdbs2:
                        # here we do not limit the residue number in each pdb, and it will be limited in model training hyper-parameters
                        # use pdb name to read the pdb file and corresponding sasa file simultaneously
                        wt_data_dict = main_chain_processing(root_path + coords_folder_path1, root_path + sasa_folder_path1, pdb_name)
                        mt_data_dict = main_chain_processing(root_path + coords_folder_path2, root_path + sasa_folder_path2, pdb_name)

                        # a check about repeated cr_token protein
                        if wt_data_dict == 0 or mt_data_dict == 0:
                            counter2 += 1 # the relevant data will not be written into json
                        else:
                            # a check about effective residue number difference between widetype and mutation
                            # because generated wt_data_dict (by here) and mt_data_dict (by manual adjustment) are both based on effective AAs
                            wt_aa_num = wt_data_dict['len_effective_aalist']
                            mt_aa_num = mt_data_dict['len_effective_aalist']
                            # check whether aa numbers of widetype and mutation are the same
                            assert wt_aa_num == mt_aa_num, 'there is a mismatching between effective WT and MT residue number, WT: {}, MT: {}'.format(wt_aa_num, mt_aa_num)
                            # currently we do not add strict check about correspondence between WT and MT side chain atoms
                            new_wt_sidec_dict, new_mt_sidec_dict = sidec_checker(pdb_name, wt_data_dict['sidechain_dict'], mt_data_dict['sidechain_dict'], mutation[2]) # mutation[2]: mutation information
                            wt_data_dict['sidechain_dict'], mt_data_dict['sidechain_dict'] = new_wt_sidec_dict, new_mt_sidec_dict

                            # besides, maybe we need to consider how to add side chain check about atom number/atom type/atom order for each side chain between WT and MT
                            # or maybe it can be solved by establishing fixed mapping between WT and MT (i.e., using the same side chain atom screening rule and ordering rule in main_chain_processing)
                            data_dict = {'name': pdb_name[:-4], 'partner': mutation[1], 'widetype_complex': wt_data_dict, 'mutation_complex': mt_data_dict,
                                         'mutation_info': interface_mutation[pdb_name[:-4]]['mutation'], 'interface': interface_mutation[pdb_name[:-4]]['interface'], 'ddg': interface_mutation[pdb_name[:-4]]['ddg']}
                            # there may be unconsistency in interface here, because interface is generated based on WT complex, which may contain some non-natural AAs,
                            # which have been removed in mutant pdbs, this can be solved in FinetuningGraphDataset when generating interface masks
                            # because FinetuningGraphDataset only gives interface mask to effective (natural) AA sites, which are shared between WT and MT complexes (based on the same cr_token list)

                            f1.write(json.dumps(data_dict) + '\n')
                            counter3 += 1

                    else:
                        # this pdb will not be written into json data source file
                        print('there is a missing/mismatching on the pdbs provided for WT and MT in {}:'.format(dataset_name1), pdb_name)
                        continue

                print('repeated cr_token protein number:', counter2)
                print('successfully generated protein number:', counter3)


# compare and screen side chain atoms based on foldx completion structures
def sidec_checker(pdb_name, wt_sidec, mt_sidec, mutation):
    wt_keys = set(wt_sidec.keys())
    common_keys = sorted(list(wt_keys))
    # mutation = [''.join([i for i in entry if i.isdigit()]) for entry in mutation.split(',')] # common digit mode
    # *** residue serial numbers in orginal mutation files include 100a/100b/100c, here we need to make them from lowercase to uppercase (i.e., 100A/100B/100C)
    mutation = ['{}_{}'.format(entry.split(':')[0], entry.split(':')[-1][1:-1].upper()) for entry in mutation.split(',')] # 'D:A488G','D:A488G' mode
    common_keys = [key for key in common_keys if key not in mutation] # in mutation sites, site chain atoms are allowed to be different

    for key in common_keys:
        # current wt_sidec[key] and mt_sidec[key] have been sorted (within main_chain_processing)
        wt_sidec_name, mt_sidec_name = np.array(wt_sidec[key])[:, 0], np.array(mt_sidec[key])[:, 0]
        wt_sidec_nameset = set(wt_sidec_name)
        mt_sidec_nameset = set(mt_sidec_name)
        if not wt_sidec_nameset == mt_sidec_nameset:
            print('there is difference between WT and MT side chain atoms in {}: {}'.format(pdb_name, key)) # allow to print, if only pass the assert check below
            # adjust mt_sidec based on wt_sidec (for current key (cr_token))
            # print(mt_sidec[key]) # [['CB', -17.401, -6.458, 72.668], ['OG', -17.883, -7.715, 72.215], ['OXT', -15.77, -3.895, 72.482]] # OXT: O of complete -COOH in a chain of the protein
            screened_mt_sidec = []
            for atom in mt_sidec[key]: # mt_sidec[key] has been sorted in json_generator
                if atom[0] in wt_sidec_nameset:
                    screened_mt_sidec.append(atom)
            mt_sidec[key] = screened_mt_sidec
            # check difference between WT and MT side chain atoms again
            assert set(np.array(wt_sidec[key])[:, 0]) == set(np.array(mt_sidec[key])[:, 0]), f'{set(np.array(wt_sidec[key])[:, 0])} | {set(np.array(mt_sidec[key])[:, 0])}'

    return wt_sidec, mt_sidec


# run this code to generate the input for pytorch Dataset
# to run this code, other the coordinate information, independent SASA files corresponding to coordinate information also need to be provided
if __name__ == '__main__':
    task = 'finetuning' # pretraining / finetuning
    dataset_name = 'M1101'
    mutant_source = 'foldx' # for specifying the mutant source for downstream tasks

    # json generator information:
    # dataset_name1 and 2 are for further determining the used dataset name, i.e., 'pretraining_set_foldx' and 'S645'/'M1101'/'S1131'/'M1707'/'S4169'

    # json_generator(task, root_path, dataset_name1, coords_folder_path1, sasa_folder_path1, dataset_name2=None, coords_folder_path2=None, sasa_folder_path2=None, save_num=None)
    if task == 'pretraining':
        # * this is the path of storing the raw coordinate and SASA files, and the generated data storage jsonl will be stored in './data/' path *
        root_path = './data/refer data/PPI/EquiPPI/data/pretraining/'
        json_generator(task, root_path, 'pretraining_set_foldx', 'pretraining_set_foldx/', 'pretraining_SASA/')

    elif task == 'finetuning':
        # * read 1_wt_pdb / 2_wt_dasa / 3_mt_pdb / 4_mt_dasa generated by previous steps for current dataset *
        # * the example of (above) folder arrangement and data structure in each folder can be found in ./data/M1101_foldx_cleaned/ *
        root_path = './data/refer data/PPI/EquiPPI/data/'
        json_generator(task, root_path, dataset_name, f'/{dataset_name}_foldx_cleaned/1_wt_pdb/', f'/{dataset_name}_foldx_cleaned/2_wt_dasa/',
                       dataset_name, f'/{dataset_name}_foldx_cleaned/3_mt_pdb_{mutant_source}/', f'/{dataset_name}_foldx_cleaned/4_mt_dasa_{mutant_source}/')
