# step1:
# this script is used for preprocessing the raw pretraining dataset, to only retain effective ATOM lines in each pdb
import os
import pickle


def cleaning_pdb(root_path, folder_name):

    # root_path = r'D:/PROJECT B2_4/data/refer data/PPI/'
    # with open('./data/all_pretraining_index.txt') as f:
    #     pdbs = f.readlines()
    pdbs = sorted(os.listdir(root_path + folder_name))

    new_path = root_path + folder_name.strip('/') + '_cleaned/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    residues = ['ARG', 'MET', 'VAL', 'ASN', 'PRO', 'THR', 'PHE', 'ASP', 'ILE', \
                'ALA', 'GLY', 'GLU', 'LEU', 'SER', 'LYS', 'TYR', 'CYS', 'HIS', 'GLN', 'TRP']
    residue_set = set(residues)
    residue_get = set()
    aa_outlier = dict()

    counter1 = 0
    for pdb in pdbs:
        counter1 += 1
        if counter1 % 1000 == 0:
            print('counter1:', counter1)

        if os.path.exists(new_path + pdb): # the cleaned pdbs have existed in new_path
            continue
        else:
            with open(root_path + folder_name + '/' + pdb) as f:
                entry = f.readlines()

            # sometimes a protein chain will interact with non-residue chain and form a complex
            # in this case, we only consider residue based chains
            screened_list = []
            counter2 = 0
            last = '0' * 20
            for i in entry:
                counter2 += 1
                resname = i[17:20].strip()

                # 18-20, 3-digit name: residue name
                # Chimera allows (nonstandard) use of four-character residue names occupying an additional column to the right (extended to 21, but 18-20 should be canonical).
                if i[0:4] == 'ATOM' and len(resname) == 3: # remove non-residue atoms
                    screened_list.append(i)
                    residue_get.add(resname)
                    if resname not in residue_set:
                        if pdb[:-4] not in aa_outlier.keys():
                            aa_outlier[pdb[:-4]] = []
                            aa_outlier[pdb[:-4]].append(resname)
                        else:
                            aa_outlier[pdb[:-4]].append(resname)
                # a check about non-residue atoms
                elif i[0:4] == 'ATOM' and len(resname) != 3:
                    print('non-residue atoms:', pdb[:-4], i)
                # foldx TER: 4-digit, pymol TER: 7-digit, pretraining dataset: 81-dight (https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html), here we unify this to 4-digit
                # raw TER: TER    1192      THR A  63
                elif i[0:3] == 'TER': # the last 'TER' will be removed after foldx
                    screened_list.append(i[0:3] + '\n')

                # for recording the last line in current pdb
                # last = i

            # currently do not change atom serial numbers and residue serial numbers
            temp_file = open(new_path + pdb, 'w')
            for i in screened_list:
                temp_file.writelines(i)
            temp_file.close()

    # store the amino acid type information for each set
    print('20 common residue set:', residue_set)
    print('residue types gotten from current dataset:', residue_get)
    with open(root_path + '{}_residue_set.pkl'.format(folder_name.strip('/')), 'wb') as f:
        pickle.dump({'residue_get':residue_get, 'aa_outlier': aa_outlier}, f)
    with open(root_path + '{}_residue_set.pkl'.format(folder_name.strip('/')), 'rb') as f:
        data = pickle.load(f)
    print('{}_residue_set.pkl:'.format(folder_name.strip('/')), data)


if __name__ == '__main__':
    root_path = './data/refer data/PPI/EquiPPI/data/' # data source folder path
    # S4169: with aa_outlier
    # M1707: without aa_outlier
    # S1131: without aa_outlier
    # M1101: without aa_outlier
    # S645: without aa_outlier
    folder_name = 'S645_foldx' # file folder to be cleaned, the processed pdbs are output to the specified fold_name with a 'cleaned' suffix
    cleaning_pdb(root_path, folder_name)







