# step2:
# *** do not interrupt this code if it is being run as it is an in_place operation ***
# We require that between two different chains, the 'TER' symbol should be added, but there should not be a 'TER' at the end of pdb files (the MpbPPI framework requirement)
# this .py script is used to unify this for the generated mutation structures
import os

# in-place 'TER' symbol adjustment
# this script assumes that other than 'TER' symbols, other conditions in generated mutation files have satisfied our requirements
def mutation_ter_unify(root_path, folder_name, use_flag):
    if use_flag:
        pdbs = sorted(os.listdir(root_path + folder_name))
        counter1 = 0

        for pdb in pdbs:
            counter1 += 1
            if counter1 % 1000 == 0:
                print('counter1:', counter1)

            with open(root_path + folder_name + pdb) as f:
                entry = f.readlines()

            # first only retain all 'ATOM' lines
            screened_list1 = []
            for line in entry:
                if line[0:4] == 'ATOM':
                    screened_list1.append(line)
            # print('screened_list1:', len(screened_list1))

            # then add the 'TER' symbols
            counter2 = 0 # record the chain number
            chain_flag = None
            screened_list2 = []
            for line in screened_list1:
                chainid = line[21]
                if chain_flag != chainid:
                    if counter2 != 0:
                        screened_list2.append('TER' + '\n')
                        counter2 += 1
                    else: # counter2 == 0
                        counter2 += 1
                    chain_flag = chainid

                screened_list2.append(line)
            # print('screened_list2:', len(screened_list2))
            print('chain number in {}: {}, TER number: {}'.format(pdb[:-4], counter2, counter2 - 1))

            temp_file = open(root_path + folder_name + pdb, 'w')
            for line in screened_list2:
                temp_file.writelines(line)
            temp_file.close()


if __name__ == '__main__':
    # *** do not interrupt this code if it is being run as it is in_place operation ***

    # path for storing WT and MT coordinate and SASA information
    root_path = 'D:/PROJECT B2_4/data/refer data/PPI/EquiPPI/data/M1101_foldx_cleaned/'
    mutation_path = '3_mt_pdb_af2/'
    # the flag for indicating whether to use this unify function
    use_flag = True
    mutation_ter_unify(root_path, mutation_path, use_flag)
