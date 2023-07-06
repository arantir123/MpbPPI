# step2:
# use foldx buildmodel to complete side chain atom
import os
import shutil
import re
import numpy as np

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
        print('total pdb number in current folder:', len(pdbs))

        # make sure every file name in pretraining_set satisfy 'suffix' + '.pdb'
        # notice a special kind of pdb name, like HM_1KTZ.pdb in S645 and M1101
        for pdb in pdbs:
            if 'ent' in pdb and not 'ent' in pdb.split('.')[0]: # "not 'ent' in pdb.split('.')[0]" is for protecting the case of 3ent_1.pdb
                shutil.move('{}/{}'.format(root_path + folder_name, pdb), '{}/{}'.format(root_path + folder_name, pdb.split('.')[0] + '.' + pdb.split('.')[-1]))
                print(pdb, '{}/{}'.format(root_path + folder_name, pdb.split('.')[0] + '.' + pdb.split('.')[-1]))
        # update the new pdb names
        pdbs = os.listdir(root_path + folder_name)

        if not os.path.exists(root_path + folder_name.strip('/') + '_foldx/'):
            os.makedirs(root_path + folder_name.strip('/') + '_foldx/')

        my_re = re.compile(r'[A-Za-z]', re.S)

        counter1 = 0
        for pdb in pdbs:
            counter1 += 1
            if counter1 % 3000 == 0:
                print('counter:', counter1)

            # for selecting specific pdbs
            # a foldx test about incomplete backbones: the incomplete residue will be deleted, and the overall residue serial number will not be re-arranged (creating a GAP)
            # if pdb != '1a41_1.pdb':
            #     continue
            print('current pdb to be processed:', pdb) # updated pdb name

            # only the successfully completed protein (through all the procedures in the loop) can be skipped
            if os.path.exists('{}/{}'.format(root_path + folder_name.strip('/') + '_foldx/', pdb)):
                print('FoldX_{} already exists'.format(pdb))
                continue

            # read current pdb
            with open(root_path + folder_name + pdb) as f:
                lines = f.readlines()
                # for line in lines:
                #     if line[0:3] == 'TER':
                #         print('length of TER line in current pdb:', len(line))

            # make sure all chain names are letters instead of numbers (for being processed by FoldX)
            # this code assumes that the obtained pdb only has 'ATOM' and 'TER' lines (in our collected pretraining dataset, 'ATOM' and 'TER' lines are all 81-digit)
            temp_set2, temp_set3 = set(), set()
            if lines[0][21].isdigit():
                # need to change the chain name
                print('The pdb file to be modified the chain name: {}'.format(pdb))
                for temp_line in lines:
                    if temp_line[0:4] == 'ATOM':
                        if (temp_line[21]).isdigit(): # temp_line[21]: getting chain id
                            temp_set2.add(chain2code[int(temp_line[21])])
                        elif (temp_line[21]).isalpha():
                            temp_set3.add(temp_line[21])

                # conflicted chain *number* id
                conflict_chains = [code2chain[i] for i in sorted(list(temp_set2.intersection(temp_set3)))]
                # for the case that all chain ids are number
                if len(conflict_chains) == 0:
                    temp_dict = chain2code
                # for the case that chain ids include number and letter
                else:
                    available_chain_names = sorted(list(set(chain_code) - temp_set3))
                    # give a new letter chain name mapping to these conflict chains
                    temp_dict = {}
                    for i, j in zip(conflict_chains, available_chain_names):
                        temp_dict[i] = j

                # then, iterate lines again for giving distinct chain names
                temp_lines = []
                for temp_line in lines:
                    if temp_line[0:4] == 'ATOM':
                        # it is possible that in a pdb file, the chain identifier includes both letter names and digit names
                        if (temp_line[21]).isdigit():
                            temp = list(temp_line)
                            temp[21] = temp_dict[int(temp_line[21])]
                            temp_line = ''.join(temp)
                        temp_lines.append(temp_line)
                    elif temp_line[0:3] == 'TER':
                        temp_lines.append(temp_line[0:3] + '\n') # here unify the 'TER' line to 4-dight, following the same rule in _1_run_cleaning_pdb.py
                # the end of this loop
                # assert len(temp_set2.intersection(temp_set3)) == 0, 'Current pdb includes error chain names (include overlapping alpha and dight chain names).'

                temp_file = open(root_path + folder_name + pdb, 'w')
                for i in temp_lines:
                    temp_file.writelines(i)
                temp_file.close()
                # record the pdb to be modified the chain names
                with open('./data/data_information/{}_chainchange_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                    f.write(pdb + '\n')
                print('chain change protein: {}'.format(pdb))
                # read the modified file again
                with open(root_path + folder_name + pdb) as f:
                    lines = f.readlines()

            counter2 = 0
            temp_set1 = set()
            for line in lines:
                # add an extra limitation, the total residue number cannot exceed 3000:
                if line[0:4] == 'ATOM' and line[12:16].strip() == 'CA':
                    counter2 +=1
                # add an extra limitation, at least in a pdb file, the backbone atoms including C, CA, N, O should be retained:
                if line[0:4] == 'ATOM':
                    temp_set1.add(line[12:16].strip()) # line[12:16].strip(): atom name
            if counter2 >= 3000:
                with open('./data/data_information/{}_overlarge_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                    f.write(pdb + '\n')
                print('overlarge protein: {}'.format(pdb))
                continue
            if len(temp_set1) < 4:  # C, CA, N, O
                with open('./data/data_information/{}_overincomplete_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                    f.write(pdb + '\n')
                print('overincomplete protein: {}'.format(pdb))
                continue

            residue_flag = -1
            temp_flag1 = False
            temp_set4 = set()
            # counter3 = 0
            # find a proper mutation site for foldx
            specified_flag = False
            if specified_flag == True:
                print('use the specified information to mutate.')
                resname, chainid, res_idx, resabbv = 'ASP', 'A', '435', 'D'
            else:
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

                                # ATOM     14  N   CYS L   1      17.664  16.833  16.160  1.00 17.75           N
                                # CYS L 1 C
                                print('resname, chainid, res_idx, resabbv:', resname, chainid, res_idx, resabbv)
                                break
            # skip current pdb if current pdb is categorized into 'aaoutlier protein'
            if temp_flag1 == True:
                continue

            # for the test example, do not forget to use exit() below to prevent false pdb generation
            # resabbv, chainid, res_idx = 'F', 'B', 13
            with open('individual_list.txt', 'w') as f:
                # cont = '{}{}{}{};'.format(wildname, chainid, resid, wildname)
                cont = '{}{}{}{};'.format(resabbv, chainid, res_idx, resabbv)
                f.write(cont) # write to the main dict for guiding foldx to complete side chain

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
                    f.write(pdb + '\n')
                print('other invalid protein: {}'.format(pdb))
                continue
            else:
                shutil.move('{}/{}_1.pdb'.format('temp', pdb[:-4]), '{}/{}'.format(root_path + folder_name.strip('/') + '_foldx/', pdb))
                print('current processed file number:', len(set(os.listdir(root_path + folder_name.strip('/') + '_foldx/'))))

            shutil.rmtree('temp')
            if not os.path.exists('temp'):
                os.makedirs('temp')

    else:
        print('required original pdb files do not exist.')


if __name__ == '__main__':
    complete_side_chain('./data/', 'pretraining_set/')