# step3:
# generate interface and mutation site information based on the order of original mutation file for downstream tasks
import os.path
import pandas as pd
import numpy as np
import json
from run_gen_interface import gen_inferface


def generating_interface_mutation(root_path, mutation_filename):
    # root_path = 'D:/PROJECT B2_4/data/refer data/PPI/EquiPPI/data/'
    # mutation_filename = 'S4169'

    if not os.path.exists(root_path + '{}_interface_mutation_dict.json'.format(mutation_filename)):

        mutation_filename = mutation_filename.strip('/')
        # S645, M1101, S1131, M1707, S4169
        # the serial number in pdb of downstream sets after foldx side chain completion starts from 1 instread of 0!
        if mutation_filename == 'M1101':
            mutation_set = pd.read_csv(root_path + mutation_filename + '.csv', encoding='latin-1')
        else:
            mutation_set = pd.read_csv(root_path + mutation_filename + '.csv')

        # foldx-processed pdb file storage folder
        temp_name = np.array([root_path + mutation_filename + '_foldx_cleaned/1_wt_pdb/', root_path + mutation_filename + '_cleaned_foldx/1_wt_pdb/'])
        foldx_folder = np.array([os.path.exists(root_path + mutation_filename + '_foldx_cleaned/1_wt_pdb/'), os.path.exists(root_path + mutation_filename + '_cleaned_foldx/1_wt_pdb/')])
        foldx_folder = temp_name[foldx_folder][0]
        print('foldx_folder:', foldx_folder)

        # (existing ddg=0 cases)
        # S645: #PDB, Partners(A_B), Protein-1, Protein-2, Mutation (D:A488G), ddG(kcal/mol), type (A or B)
        # M1101: #PDB, Partners(A_B), Protein-1, Protein-2, Mutation (D:L483T,D:V486P,D:H487A,D:A488M,D:I491L,D:A492P,D:M496I), ddG(kcal/mol)
        # S1131: protein, Partners(A_B), mutation (A:C171A), DDG, mode (forward)
        # M1707: PDB id, Partner1, Mutation(s)_PDB (DA11A,MA14V,HA18Q,RA19H,FA25A,QA29K,EA33R), Mutation(s)_cleaned, DDGexp, Label (forward), SA_com_wt, SA_part_wt
        # S4169: protein, Partners(A_B), mutation (A:D8A), DDG, mode (forward)
        if mutation_filename == 'S645':
            mutation_set = mutation_set[['#PDB', 'Partners(A_B)', 'Mutation', 'ddG(kcal/mol)']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
        elif mutation_filename == 'M1101':
            mutation_set = mutation_set[['#PDB', 'Partners(A_B)', 'Mutation', 'ddG(kcal/mol)']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
        elif mutation_filename == 'S1131':
            mutation_set = mutation_set[['protein', 'Partners(A_B)', 'mutation', 'DDG']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
        elif mutation_filename == 'M1707':
            mutation_set = mutation_set[['PDB id', 'Partner1', 'Mutation(s)_cleaned', 'DDGexp']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
            # for M1707, for consistency, change its mutation information to 'D:A488G' mode
            # The mutation information includes WT residue, chain, residue index and mutant residue. such as “TI38F”
            # which stands for mutating the 38th acid amino at the I chain (i.e., phenylalanine) to threonine.
            mutation_ = []
            for row in mutation_set['mutation']:
                temp = ','.join([i[1] + ':' + i[0] + i[2:-1] + i[-1] for i in row.split(',')])
                mutation_.append(temp)
            mutation_set['mutation'] = mutation_
            # print(mutation_set)
        elif mutation_filename == 'S4169':
            mutation_set = mutation_set[['protein', 'Partners(A_B)', 'mutation', 'DDG']]
            mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']

        mutation_set_np = np.array(mutation_set) # pdb, partner, mutation, ddg

        workdir = 'temp' # the path to store temporary interface file from pymol
        counter1 = 0
        interface_mutation_dict = dict()
        # generate information of all mutations in the specified downstream dataset
        for entry in mutation_set_np:
            counter1 += 1
            if counter1 % 500 == 0:
                print('counter:', counter1)

            name = entry[0] + '__' + str(counter1) + '.pdb'
            if_info = entry[1] # partner information

            # for pymol, if no special operations are conducted, it will not change the atom number and corresponding coordinates
            # however, it will:
            # 1) add pure element type into each pdb line (the last part),
            # 2) renumber atom serial number (from 1, not changing residue serial number),
            # 3) arrange atom order in the same residue
            # and add 7-digit 'TER' and 'END' to the end of the pdb file

            if os.path.exists(foldx_folder + name):
                interfaces = gen_inferface(foldx_folder, name, if_info, workdir) # output to interface.txt
            else:
                continue

            if name[:-4] not in interface_mutation_dict.keys():
                interface_mutation_dict[name[:-4]] = {'name': name[:-4], 'partner': if_info, 'mutation': entry[2], 'ddg': entry[3], 'interface': interfaces}

        with open(root_path + '{}_interface_mutation_dict.json'.format(mutation_filename), 'w') as outfile:
            json.dump(interface_mutation_dict, outfile)

    else:
        print(root_path + '{}_interface_mutation_dict.json has already exists'.format(mutation_filename))


if __name__ == '__main__':
    root_path = './data/refer data/PPI/EquiPPI/data/'
    mutation_filename = 'S645'
    generating_interface_mutation(root_path, mutation_filename)