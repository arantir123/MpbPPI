# the script for directly running foldx to predict PPI ddG, this script can also be used to summarize results output by _4_run_finetuning_model_*.py
# for fair comparison, it is better to run this script based on the widetype complexes after side chain completion function (i.e., _2_run_completing_sidechain_downstream.py)
# this could make all methods start the mutation simulation based on the same WT structures
import os
import pandas as pd
import numpy as np
import shutil
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats
import re


def foldx_mutation_prediction(root_path, folder_name, only_testing=None):
    # hyper-parameters: root_path, folder_name, only_testing=None, in which folder_name refers to the folder storing the above 4 sub-folders
    # only_testing is for finding and calculating binding affinity change of one sample in the original mutation file (start from 1, based on the original sample order)
    if not os.path.exists('temp'):
        os.makedirs('temp')
    # save the output from foldx
    if not os.path.exists(root_path + folder_name.strip('/') + '/foldx_mutant_structure/'):
        os.makedirs(root_path + folder_name.strip('/') + '/foldx_mutant_structure/')
    if not os.path.exists(root_path + folder_name.strip('/') + '/foldx_mutant_Average/'):
        os.makedirs(root_path + folder_name.strip('/') + '/foldx_mutant_Average/')
    if not os.path.exists(root_path + folder_name.strip('/') + '/foldx_mutant_Dif/'):
        os.makedirs(root_path + folder_name.strip('/') + '/foldx_mutant_Dif/')
    if not os.path.exists(root_path + folder_name.strip('/') + '/foldx_mutant_Raw/'):
        os.makedirs(root_path + folder_name.strip('/') + '/foldx_mutant_Raw/')
    if not os.path.exists(root_path + folder_name.strip('/') + '/foldx_mutant_PdbList/'):
        os.makedirs(root_path + folder_name.strip('/') + '/foldx_mutant_PdbList/')

    print('current pdb coordinate and SASA folder root:', root_path + folder_name)
    # original mutation file is currently stored in './data/'
    if folder_name.strip('/').split('_')[0] == 'M1101':
        mutation_set = pd.read_csv('./data/' + 'M1101.csv', encoding='latin-1')
        mutation_set = mutation_set[['#PDB', 'Partners(A_B)', 'Mutation', 'ddG(kcal/mol)']]
        mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
    elif folder_name.strip('/').split('_')[0] == 'M1707':
        mutation_set = pd.read_csv('./data/' + 'M1707.csv')
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
    elif folder_name.strip('/').split('_')[0] == 'S645':
        mutation_set = pd.read_csv('./data/' + 'S645.csv')
        mutation_set = mutation_set[['#PDB', 'Partners(A_B)', 'Mutation', 'ddG(kcal/mol)']]
        mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
    elif folder_name.strip('/').split('_')[0] == 'S1131':
        mutation_set = pd.read_csv('./data/' + 'S1131.csv')
        mutation_set = mutation_set[['protein', 'Partners(A_B)', 'mutation', 'DDG']]
        mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
    elif folder_name.strip('/').split('_')[0] == 'S4169':
        mutation_set = pd.read_csv('./data/' + 'S4169.csv')
        mutation_set = mutation_set[['protein', 'Partners(A_B)', 'mutation', 'DDG']]
        mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
    mutation_set_np = np.array(mutation_set) # 'pdb', 'partner', 'mutation', 'ddg'

    # only calculate one pdb results based on the given sample id
    if only_testing:
        print('original mutation file sample:', mutation_set_np[only_testing-1, :])
        pdbs = [mutation_set_np[only_testing-1, :][0] + '__' + str(only_testing) + '.pdb']
    # get all pdbs in current dataset (for creating the results of all mutants)
    else:
        pdbs = sorted(os.listdir(root_path + folder_name.strip('/') + '/1_wt_pdb/'))
        print('total pdb wide-type complex number in current widetype complex folder:', len(pdbs))

    # modelling mutations based on above defined pdbs
    counter1 = 0
    for pdb in pdbs:
        counter1 += 1
        if counter1 % 500 == 0:
            print('counter:', counter1)

        # pdb example: 1AK4__1.pdb
        sample_id = int(pdb[:-4].split('__')[-1])
        name, partner, mutation, ddg = mutation_set_np[sample_id - 1, :]
        # print(mutation_set_np.shape) # (645, 4)

        # only the successfully completed protein (through all the procedures in the loop) can be skipped
        if os.path.exists('{}/{}'.format(root_path + folder_name.strip('/') + '/foldx_mutant_structure/', pdb)):
            print('FoldX_{} already exists in {}'.format(pdb, root_path + folder_name.strip('/') + '/foldx_mutant_structure/'))
            continue

        # record the mutation information
        cont = ''
        counter2 = 0
        mutation = mutation.split(',')
        invalid_mutation_list = []
        # the mutation information for M1707 has been formatted above (i.e., has been changed to 'D:A488G,D:A488G' mode)
        for item in mutation:
            counter2 += 1
            item = item.split(':')
            chainid = item[0]
            resabbv = item[-1][0]
            res_idx = item[-1][1:-1]
            mutname = item[-1][-1]
            if bool(re.search(r'[a-zA-Z]', res_idx)):
                invalid_mutation_list.append([chainid, resabbv, res_idx, mutname])
            if counter2 < len(mutation):
                cont += '{}{}{}{},'.format(resabbv, chainid, res_idx, mutname)
            else:
                cont += '{}{}{}{};'.format(resabbv, chainid, res_idx, mutname)
        # print(pdb, cont, name, partner, mutation, ddg)
        # 1JRH__138.pdb AI103L,VI104M,RI106L,DI107K; 1JRH HL_I ['I:A103L', 'I:V104M', 'I:R106L', 'I:D107K'] 0.64

        # need to do a further operation here to handle pdb like 1N8Z.pdb which contains residue names 100A+100B+100C+100D, which cannot be identified by foldx (numerical AA id only)
        if len(invalid_mutation_list) > 0:
            print('current pdb {} contained invalid mutation information which cannot identified by foldx: {}'.format(pdb, invalid_mutation_list))
            # this part of code assumes that the pdb format has been changed to our defined format (without the last 'TER')
            residue_flag, original_crtoken_record, original_crtoken_record_reverse, counter3, counter4 = -100, [dict()], [dict()], 0, 1 # for giving new serial number
            cr_token_list, chainid2counter = [], dict()
            with open(root_path + folder_name.strip('/') + '/1_wt_pdb/' + pdb) as f:
                lines = f.readlines()
            for line in lines:
                if line[0:4] == 'ATOM':
                    chainid = line[21] # chain id
                    res_idx = line[22:27].strip() # current residue idx, 100A
                    cr_token = '{}_{}'.format(chainid, res_idx)
                    if residue_flag != cr_token:
                        original_crtoken_record[counter3][res_idx] = counter4 # counter4 is the newly given AA serial number
                        original_crtoken_record_reverse[counter3][str(counter4)] = res_idx
                        residue_flag = cr_token
                        cr_token_list.append(cr_token)
                        counter4 += 1
                    chainid2counter[str(chainid)] = counter3 # assuming chain id is unique in the pdb file
                elif line[0:3] == 'TER':
                    original_crtoken_record.append(dict())
                    original_crtoken_record_reverse.append(dict())
                    counter3 += 1
                    counter4 = 1 # restart to give AA serial number
            # the check is for checking whether the newly given id number equals to the original cr_token number
            assert len(cr_token_list) == sum([len(list(i.keys())) for i in original_crtoken_record]), '{}, {}'.format(len(cr_token_list), sum([len(list(i.keys())) for i in original_crtoken_record]))

            # get updated pdb and store it in 'temp' root
            update_pdb = []
            counter5 = 0
            for line in lines:
                if line[0:4] == 'ATOM':
                    res_idx = line[22:27].strip()  # current residue idx, 100A
                    new_number = str(original_crtoken_record[counter5][res_idx])
                    line = list(line) # transform line to a list
                    # clean the original res_idx
                    # if ''.join(line[23: 27]) == '100A':
                    #     print(line[22: 27])
                    line[22: 27] = ' ', ' ', ' ', ' ', ' '
                    # print(line[22: 27])
                    line[26 - len(new_number): 26] = new_number # only occupy 23-26 (starting from 1)
                    line = ''.join(line)
                    update_pdb.append(line)
                elif line[0:3] == 'TER':
                    update_pdb.append(line)
                    counter5 += 1
                else:
                    update_pdb.append(line)
            temp_file = open('./temp/' + pdb, 'w')
            for i in update_pdb:
                temp_file.writelines(i)
            temp_file.close()

            # cont update
            cont = ''
            counter6 = 0
            for item in mutation:
                counter6 += 1
                item = item.split(':')
                chainid = item[0]
                resabbv = item[-1][0]
                res_idx = item[-1][1:-1]
                mutname = item[-1][-1]
                if counter6 < len(mutation):
                    # res_idx.upper() is in case that '100a' is not same to '100A'
                    # print(resabbv, chainid, original_crtoken_record[chainid2counter[str(chainid)]][res_idx.upper()], mutname)
                    cont += '{}{}{}{},'.format(resabbv, chainid, original_crtoken_record[chainid2counter[str(chainid)]][res_idx.upper()], mutname)
                else:
                    # print(resabbv, chainid, original_crtoken_record[chainid2counter[str(chainid)]][res_idx.upper()], mutname)
                    cont += '{}{}{}{};'.format(resabbv, chainid, original_crtoken_record[chainid2counter[str(chainid)]][res_idx.upper()], mutname)

        print('cont for mutation:', cont)
        # add a checkpoint for further adjusting cout
        # cont = 'YH100AY'
        with open('individual_list.txt', 'w') as f:
            f.write(cont)

        if len(invalid_mutation_list) > 0:
            comm = 'foldx_4 --command=BuildModel --pdb={}  --mutant-file={} --output-dir={} --pdb-dir={}'.format(pdb, 'individual_list.txt', 'temp', 'temp')
        else:
            # * print(root_path + folder_name.strip('/') + '/1_wt_pdb/') ./data/M1101_foldx_cleaned/1_wt_pdb/, path to store WT PDB files to be mutated *
            comm = 'foldx_4 --command=BuildModel --pdb={}  --mutant-file={} --output-dir={} --pdb-dir={}'.format(pdb, 'individual_list.txt', 'temp', root_path + folder_name.strip('/') + '/1_wt_pdb/')
        os.system(comm)
        print('finishing to run FoldX.')

        # * the case of FoldX mutant structure generation failure (record and skip this PDB name, and start to process the next PDB) *
        if not os.path.exists('{}/{}_1.pdb'.format('temp', pdb[:-4])): # pdb example: 1AK4__1.pdb
            # you need to establish this folder to store problematic PDBs based on your specified folder name
            with open('./data/data_information/{}_otherinvalid_protein.txt'.format(folder_name.strip('/')), 'a') as f:
                f.write(pdb + '\n')
            print('other invalid protein: {}'.format(pdb))
            continue
        # * the case of FoldX mutant structure generation success (store the successfully generated mutant structure to 'foldx_mutant_structure' folder) *
        else:
            if len(invalid_mutation_list) > 0:
                # give the original AA serial number back for unifying
                with open('{}/{}_1.pdb'.format('temp', pdb[:-4])) as f:
                    lines = f.readlines() # assuming the total residue number will not change in current task

                reverse_pdb = []
                counter7 = 0
                for line in lines:
                    if line[0:4] == 'ATOM':
                        res_idx = line[22:27].strip()
                        # original_crtoken_record_reverse[counter3][str(counter4)] = res_idx, res_idx orginates from 23-27 dights (starting from 1)
                        new_number = str(original_crtoken_record_reverse[counter7][res_idx]) # current new_number could be '100A' from 23-27 dights (starting from 1)
                        line = list(line)  # transform line to a list
                        # clean the original res_idx from 23-27 dights (starting from 1)
                        line[22: 27] = ' ', ' ', ' ', ' ', ' '
                        if bool(re.search(r'[a-zA-Z]', new_number)): # containing letter
                            line[26] = new_number[-1]
                            line[26 - len(new_number[:-1]): 26] = new_number[:-1]  # occupy 23-26 (starting from 1)
                        else:
                            line[26 - len(new_number): 26] = new_number
                        line = ''.join(line)
                        reverse_pdb.append(line)
                    elif line[0:3] == 'TER':
                        reverse_pdb.append(line)
                        counter7 += 1
                    else:
                        reverse_pdb.append(line)
                temp_file = open('{}/{}'.format(root_path + folder_name.strip('/') + '/foldx_mutant_structure/', pdb), 'w')
                # temp_file = open('{}/{}'.format('temp', pdb), 'w')
                for i in reverse_pdb:
                    temp_file.writelines(i)
                temp_file.close()
            else:
                shutil.move('{}/{}_1.pdb'.format('temp', pdb[:-4]), '{}/{}'.format(root_path + folder_name.strip('/') + '/foldx_mutant_structure/', pdb))

            shutil.move('{}/{}.fxout'.format('temp', 'Average_' + pdb[:-4]), '{}/{}'.format(root_path + folder_name.strip('/') + '/foldx_mutant_Average/', '{}.fxout'.format('Average_' + pdb[:-4])))
            shutil.move('{}/{}.fxout'.format('temp', 'Dif_' + pdb[:-4]), '{}/{}'.format(root_path + folder_name.strip('/') + '/foldx_mutant_Dif/', '{}.fxout'.format('Dif_' + pdb[:-4])))
            shutil.move('{}/{}.fxout'.format('temp', 'PdbList_' + pdb[:-4]), '{}/{}'.format(root_path + folder_name.strip('/') + '/foldx_mutant_PdbList/', '{}.fxout'.format('PdbList_' + pdb[:-4])))
            shutil.move('{}/{}.fxout'.format('temp', 'Raw_' + pdb[:-4]), '{}/{}'.format(root_path + folder_name.strip('/') + '/foldx_mutant_Raw/', '{}.fxout'.format('Raw_' + pdb[:-4])))
            print('current processed file number:', len(set(os.listdir(root_path + folder_name.strip('/') + '/foldx_mutant_structure/'))))

        shutil.rmtree('temp')
        if not os.path.exists('temp'):
            os.makedirs('temp')


def foldx_output_summary(dataset_name, output_path):

    print('current foldx Dif file folder root:', output_path)
    # original mutation file is currently stored in './data/'
    if dataset_name == 'M1101':
        mutation_set = pd.read_csv('./data/' + 'M1101.csv', encoding='latin-1')
        mutation_set = mutation_set[['#PDB', 'Partners(A_B)', 'Mutation', 'ddG(kcal/mol)']]
        mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
    elif dataset_name == 'M1707':
        mutation_set = pd.read_csv('./data/' + 'M1707.csv')
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
    elif dataset_name == 'S645':
        mutation_set = pd.read_csv('./data/' + 'S645.csv')
        mutation_set = mutation_set[['#PDB', 'Partners(A_B)', 'Mutation', 'ddG(kcal/mol)']]
        mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
    elif dataset_name == 'S1131':
        mutation_set = pd.read_csv('./data/' + 'S1131.csv')
        mutation_set = mutation_set[['protein', 'Partners(A_B)', 'mutation', 'DDG']]
        mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
    elif dataset_name == 'S4169':
        mutation_set = pd.read_csv('./data/' + 'S4169.csv')
        mutation_set = mutation_set[['protein', 'Partners(A_B)', 'mutation', 'DDG']]
        mutation_set.columns = ['pdb', 'partner', 'mutation', 'ddg']
    mutation_set_np = np.array(mutation_set) # 'pdb', 'partner', 'mutation', 'ddg'

    Difs = sorted(os.listdir(output_path))
    counter = 0
    predictions, labels = [], []
    for dif in Difs:
        counter += 1
        if counter % 500 == 0:
            print('counter:', counter)

        dif_ = int(dif[:-6].split('__')[-1])
        with open(output_path.strip('/') + '/' + dif) as f:
            prediction = float(f.readlines()[-1].split()[1])
        predictions.append(prediction)
        labels.append(float(mutation_set_np[dif_ - 1, :][-1]))
    predictions, labels = np.array(predictions).reshape(-1), np.array(labels).reshape(-1)
    MSE = mean_squared_error(labels, predictions)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(labels, predictions)
    PEARSON = scipy.stats.pearsonr(labels, predictions)[0]

    print('FoldX prediction for {} dataset: MSE:'.format(dataset_name), MSE, 'RMSE:', RMSE, 'MAE:', MAE, 'Pearson:', PEARSON)


if __name__ == '__main__':
    # step1: calling foldx (mode: calling)
    # step2: summary the foldx output (mode: summary)
    mode = 'calling'

    if mode == 'calling':
        # * should be based on the defined standard folder format (with 4 WT/MT coordinate and SASA sub-folders), i.e., 1_wt_pdb, 2_wt_dasa, 3_mt_pdb, 4_mt_dasa *
        # * hyper-parameters: root_path, folder_name, only_testing=None, in which folder_name refers to the folder storing the above 4 sub-folders *
        # * only_testing is for finding and calculating binding affinity change of one sample in the original mutation file (start from 1, based on the original sample order) *
        foldx_mutation_prediction('./data/', 'M1101_foldx_cleaned/', 139)
    elif mode == 'summary':
        dataset_name = 'S645'
        # path to store foldx Dif files (i.e., ddG prediction files)
        foldx_output_path = './data/refer data/PPI/EquiPPI/data/{}_foldx_cleaned/foldx_mutant_Dif/'.format(dataset_name)
        foldx_output_summary(dataset_name, foldx_output_path)
