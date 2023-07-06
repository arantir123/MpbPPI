# step2:
# use foldx buildmodel to complete side chain atom
# for downstream pdbs, non-ATOM entries like HETATM have not been removed from each pdb file
# after the side chain completion, we need consider how to correspond a foldx WT protein with AF2 generated MT protein (and removing all non-ATOM entries)
import os
import shutil
import re
import numpy as np
import pandas as pd


def complete_side_chain(root_path, folder_name):
    if not os.path.exists('temp'):
        os.makedirs('temp')

    print('root_path + folder_name:', root_path + folder_name)
    # in order to generate effective residue graph for equivariant models, all residues in a protein should have complete backbone atoms (C, CA, N, O)
    if os.path.exists(root_path + folder_name):
        residues = ['ARG', 'MET', 'VAL', 'ASN', 'PRO', 'THR', 'PHE', 'ASP', 'ILE',
                    'ALA', 'GLY', 'GLU', 'LEU', 'SER', 'LYS', 'TYR', 'CYS', 'HIS', 'GLN', 'TRP']
        res_code = ['R', 'M', 'V', 'N', 'P', 'T', 'F', 'D', 'I',
                    'A', 'G', 'E', 'L', 'S', 'K', 'Y', 'C', 'H', 'Q', 'W']
        res2code = {x: idxx for x, idxx in zip(residues, res_code)}
        chains = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        chain_code = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'G', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        chain2code = {x: idxx for x, idxx in zip(chains, chain_code)}
        code2chain = {x: idxx for x, idxx in zip(chain_code, chains)}

        # read the pdb name set in current folder
        pdbs = os.listdir(root_path + folder_name)
        print('total pdb wide-type complex number in current folder:', len(pdbs))
        # read the original mutation file for completing side chain atoms more precisely, S stands for single site mutation, while M stands for muliple site mutation
        if folder_name.strip('./') == 'M1101':
            mutation_set = pd.read_csv(root_path + 'M1101.csv', encoding='latin-1')
        elif folder_name.strip('./') == 'M1707':
            mutation_set = pd.read_csv(root_path + 'M1707.csv')
        elif folder_name.strip('./') == 'S641':
            mutation_set = pd.read_csv(root_path + 'S641.csv')
        elif folder_name.strip('./') == 'S645':
            mutation_set = pd.read_csv(root_path + 'S645.csv')
        elif folder_name.strip('./') == 'S1131':
            mutation_set = pd.read_csv(root_path + 'S1131.csv')
        elif folder_name.strip('./') == 'S4169':
            mutation_set = pd.read_csv(root_path + 'S4169.csv')
        mutation_set_np = np.array(mutation_set)

        if not os.path.exists(root_path + folder_name.strip('/') + '_foldx/'):
            os.makedirs(root_path + folder_name.strip('/') + '_foldx/')
        my_re = re.compile(r'[A-Za-z]', re.S)

        # for downstream side chain completion, there are some complexes which are very big, not suitable to be completed repeatedly per mutation case
        # in this case, a name can be set, for all mutations based on the same widetype complex, only one completed pdb will be used as the widetype protein for all mutation cases
        large_complex_name = '1KBH'

        counter1 = 0 # counter1 is used to record which pdb is being processed now in downstream sets (the order is based on corresponding csv files, i.e., mutation_set_np)
        for i in range(mutation_set_np.shape[0]):
            counter1 += 1
            if counter1 % 3000 == 0:
                print('counter:', counter1)

            current_entry = mutation_set_np[i]
            pdb = current_entry[0] + '.pdb'

            # for selecting specific pdbs
            # a foldx test about incomplete backbones: the incomplete residue will be deleted, and the overall residue serial number will not be re-arranged
            # if pdb != '1JRH.pdb' or counter1 != 138:
            #     continue
            print('current pdb to be processed:', pdb, 'counter:', counter1) # updated pdb name

            # only the successfully completed protein (through all the procedures in the loop) can be skipped
            if os.path.exists('{}/{}'.format(root_path + folder_name.strip('/') + '_foldx', pdb[:-4] + '__' + str(counter1) + '.pdb')):
                print('FoldX_{} already exists'.format(pdb))
                continue

            # before reading current pdb for finding side chain completion positions and calling foldx
            # first to check large_complex_name for saving the completion time for over large proteins
            if large_complex_name and large_complex_name == current_entry[0]: # current_entry[0] does not include '.pdb'
                existing_pdbs = os.listdir('{}/'.format(root_path + folder_name.strip('/') + '_foldx'))
                temp_list = sorted([i for i in existing_pdbs if (large_complex_name in i)])
                if len(temp_list) != 0: # only in this case, the previously generated pdb will be copied, otherwise still need to call foldx
                    shutil.copyfile('{}/{}'.format(root_path + folder_name.strip('/') + '_foldx', temp_list[0]),
                                    '{}/{}'.format(root_path + folder_name.strip('/') + '_foldx', pdb[:-4] + '__' + str(counter1) + '.pdb'))
                    continue

            # read current pdb
            with open(root_path + folder_name + pdb) as f:
                lines = f.readlines()

            # a check to make sure all chain names are letters instead of numbers (for being processed by FoldX)
            if lines[0][21].isdigit(): # get the chain id of the first line of pdb (should be an 'ATOM' line)
                # record the pdb that should be modified the chain names
                with open('./data/data_information/{}_chainchange_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                    f.write(pdb[:-4] + '__' + str(counter1) + '.pdb' + '\n')
                print('chain change protein: {}'.format(pdb[:-4] + '__' + str(counter1) + '.pdb'))
                # *** when processing the downstream datasets, current plan is just to record such samples, instead of modifying them like the way of completing the pretraining set ***
                continue # skip to the next pdb

            counter2 = 0
            temp_set1 = set()
            for line in lines:
                # add an extra limitation, the total residue number cannot exceed 3000:
                if line[0:4] == 'ATOM' and line[12:16].strip() == 'CA':
                    counter2 +=1
                # add an extra limitation, at least in a pdb file, the backbone atoms including C, CA, N, O should be retained:
                if line[0:4] == 'ATOM':
                    temp_set1.add(line[12:16].strip())
            if counter2 >= 3000:
                with open('./data/data_information/{}_overlarge_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                    f.write(pdb[:-4] + '__' + str(counter1) + '.pdb' + '\n')
                print('overlarge protein: {}'.format(pdb[:-4] + '__' + str(counter1) + '.pdb'))
                continue
            if len(temp_set1) < 4:  # C, CA, N, O
                with open('./data/data_information/{}_overincomplete_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                    f.write(pdb[:-4] + '__' + str(counter1) + '.pdb' + '\n')
                print('overincomplete protein: {}'.format(pdb[:-4] + '__' + str(counter1) + '.pdb'))
                continue

            # find a proper mutation site for foldx
            residue_flag = -1
            temp_flag1 = False
            temp_set4 = set()
            specified_flag = True # for the case of that the mutation information is known (for downstream datasets)
            if specified_flag == True:
                # M1101, S641, S645, S1131, S4169 mutation information is on 5th column with the format of D:L483T
                # M1707 mutation information is on 5th column with the format of DA11A
                cont = ''
                counter3 = 0
                mutations = current_entry[4].split(',') # DA11A,MA14V,HA18Q,RA19H,FA25A,QA29K,EA33R

                # foldx cont example: 'FA39L,FB39L;'
                for mutation in mutations:
                    counter3 += 1
                    if folder_name.strip('./') == 'M1707': # DA11A,MA14V,HA18Q,RA19H,FA25A,QA29K,EA33R
                        resabbv = mutation[0] # WT
                        chainid = mutation[1]
                        res_idx = mutation[2:-1]
                        mutname = mutation[-1]
                    else: # D:L483T,D:V486P,D:H487A,D:A488M,D:I491L,D:A492P,D:M496I
                        mutation = mutation.split(':')
                        chainid = mutation[0]
                        resabbv = mutation[-1][0]
                        res_idx = mutation[-1][1:-1]
                        mutname = mutation[-1][-1]

                    if counter3 < len(mutations):
                        cont += '{}{}{}{},'.format(resabbv, chainid, res_idx, resabbv)
                    else:
                        cont += '{}{}{}{};'.format(resabbv, chainid, res_idx, resabbv)

                print('cout for mutation:', cont)
                # add a checkpoint for further adjusting cout
                # cont = 'YH100AY'
                with open('individual_list.txt', 'w') as f:
                    f.write(cont)

            else: # for the pretraining set and other sets
                for line in lines:
                    # print(line[0:4], line[22:28].strip(), re.findall(my_re, line[22:28].strip()), re.findall(my_re, '1a'))
                    # it seems that foldx cannot handle the residue serial number like 1A/1B (based on line[22:28]), thus filter it out
                    # it seems that foldx cannot handle the mutation on residue that is lack of complete backbone atoms (C/CA/N/O), becuase foldx will remove this residue (Creating a GAP, e.g., 1hxn_1.pdb)
                    if line[0:4] == 'ATOM':
                        # res_idx = int(line[22:28].strip()) # residue serial number
                        res_idx = line[22:28].strip() # need to consider residue serial number like 1A/1B, thus cannot use int transformation here
                        if residue_flag != res_idx:
                            # check the logic of foldx for processing incomplete backbone atoms
                            # if not ('C' in temp_set4 and 'CA' in temp_set4 and 'N' in temp_set4 and 'O' in temp_set4) and counter3 > 0:
                            #     print('residue_flag:', residue_flag, line, pdb)
                            #     exit()
                            residue_flag = res_idx
                            temp_set4 = set()
                            temp_set4.add(line[12:16].strip()) # atom name
                            # counter3 += 1
                        else:
                            temp_set4.add(line[12:16].strip())

                        if 'C' in temp_set4 and 'CA' in temp_set4 and 'N' in temp_set4 and 'O' in temp_set4: # need to find a residue postion in which backbone atoms are complete

                            if not len(re.findall(my_re, line[22:28].strip())): # find residue serial number not like 1A/1B
                                resname = line[17:21].strip() # had been modified, actually residue is three-digit rather than 4- or more digit, like ALA
                                chainid = line[21] # A
                                res_idx = line[22:28].strip() # 4 (is a range, info[2:-1])

                                # need to consider manual-made amino acid type
                                if resname in res2code.keys():
                                    resabbv = res2code[resname]
                                else:
                                    # if outlier residue type is found, this protein will not be retained (resabbv as the input of foldx will not be generated)
                                    with open('./data/data_information/{}_aaoutlier_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                                        f.write(pdb + '\n')
                                    print('aaoutlier protein: {}'.format(pdb))
                                    temp_flag1 = True
                                    break

                                print('resname, chainid, res_idx, resabbv:', resname, chainid, res_idx, resabbv)
                                break

                # for the test example, do not forget to use exit() below to prevent false pdb generation
                # resabbv, chainid, res_idx = 'F', 'B', 13
                with open('individual_list.txt', 'w') as f:
                    # cont = '{}{}{}{};'.format(wildname, chainid, resid, wildname)
                    cont = '{}{}{}{};'.format(resabbv, chainid, res_idx, resabbv)
                    f.write(cont)  # write to the main dict for guiding foldx to complete side chain
            # skip current pdb if current pdb is categorized into 'aaoutlier protein'
            if temp_flag1 == True:
                continue

            # call foldx
            # pdb is the full pdb file name
            comm = 'foldx_4 --command=BuildModel --pdb={}  --mutant-file={} --output-dir={} --pdb-dir={}'.format(
                pdb, 'individual_list.txt', 'temp', root_path + folder_name)  # output the generated files to temp folder
            os.system(comm)
            print('finishing to run FoldX.')
            # exit() # end here for observing the output of foldx

            # if code normally runs to here, indicating that at least no explicit error arises during calling foldx (but it is still possible that no valid mutation file is generated)
            # pdb[:-4] is the pdb file name removing the '.pdb' suffix
            # record pdbs that cannot be processed by foldx due to the reasons beyond all the screening logics above
            if not os.path.exists('{}/{}_1.pdb'.format('temp', pdb[:-4])):
                with open('./data/data_information/{}_otherinvalid_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                    f.write(pdb[:-4] + '__' + str(counter1) + '.pdb' + '\n')
                print('other invalid protein: {}'.format(pdb[:-4] + '__' + str(counter1) + '.pdb'))
                continue
            else:
                shutil.move('{}/{}_1.pdb'.format('temp', pdb[:-4]), '{}/{}'.format(root_path + folder_name.strip('/') + '_foldx/', pdb[:-4] + '__' + str(counter1) + '.pdb'))
                print('current processed file number:', len(set(os.listdir(root_path + folder_name.strip('/') + '_foldx/'))))

            shutil.rmtree('temp')
            if not os.path.exists('temp'):
                os.makedirs('temp')

    else:
        print('required original pdb files do not exist.')


if __name__ == '__main__':
    # for downstream pdbs, non-ATOM entries like HETATM have not been removed from each pdb file
    # after the side chain completion, we need consider how to correspond a foldx WT protein with AF2 generated MT protein (and removing all non-ATOM entries)
    complete_side_chain('./data/', 'M1101/') # the processed pdbs are output to the specified fold_name with an extra 'foldx' suffix