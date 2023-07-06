# step3:
# data split for downstream set (training-validation)
import os.path
import random
import json
import numpy as np
import pandas as pd


# 1, 256, 429, 512, 1002, 1012, 1024
random_seed = 512
random.seed(random_seed)
np.random.seed(random_seed)
print('random_seed:', random_seed)

# *** split_mode == 'identity' and criteria4identity == 'strict' and split_mode == 'identity' and criteria4identity == 'noval' have not used data_split_for_training ***
def data_split_for_training(all_pretraining_index, folds, train_fold, val_fold, test_fold=None):

    prng = np.random.RandomState(random_seed)
    allindex = prng.permutation(len(all_pretraining_index))
    pos_inter_fold = np.array_split(allindex, folds)

    train_pos_sample = [pos_inter_fold[i] for i in train_fold]
    val_pos_sample = [pos_inter_fold[i] for i in val_fold]
    if test_fold:
        test_pos_sample = [pos_inter_fold[i] for i in test_fold]

    # train_pos_sample: array type
    train_pos_sample = np.concatenate(train_pos_sample)
    val_pos_sample = np.concatenate(val_pos_sample)
    if test_fold:
        test_pos_sample = np.concatenate(test_pos_sample)

    # sorted: list type
    if test_fold:
        return np.array(sorted(train_pos_sample)), np.array(sorted(val_pos_sample)), np.array(sorted(test_pos_sample))
    else:
        return np.array(sorted(train_pos_sample)), np.array(sorted(val_pos_sample))


def get_proteinsplit_for_evaluation(identity_path, identity_threshold):
    # notice the unit cell format in the original csv (for col and row indices, should be text in case examples like '1E96' being transformed into '1.00E+96')
    seq_indentity = pd.read_csv(identity_path, index_col=0)
    WT_index = np.array(seq_indentity.index)
    WT_number = len(WT_index)
    seq_indentity_np = np.array(seq_indentity) - np.eye(WT_number)
    print('total WT complex number:', WT_number)

    val_test_id = []
    for i in range(WT_number):
        line = seq_indentity_np[i, :]
        if (line <= identity_threshold).all():
            val_test_id.append(i)
    val_test_id = np.array(val_test_id)
    val_test_name = sorted(list(WT_index[val_test_id]))
    train_name = sorted(list(set(WT_index) - set(val_test_name)))
    print('complex types in training set:', len(train_name), 'in val_test set:', len(val_test_name))

    return train_name, val_test_name


if __name__ == '__main__':
    # the name of the set to be split
    set_name = 'M1101'
    # split_mode: 'CV10_random' (random allocation to training/val/test sets), 'complex' (allocation based on complex), 'identity' (allocation based on sequence identity, e.g., 40%), 'plotting'
    split_mode = 'complex'
    # the path to read the jsonl source data file produced by _3_generate_json_input
    source_folder = './data/'
    # for indicating the source where mutation structures come from
    mutation_source = '_foldx'
    # splitting file output path: './data/'

    if split_mode == 'identity':
        # extra criteria for split_mode = 'identity' (criteria4identity), choices: 'strict', 'loose', and 'noval'
        # 'strict': get the val_test set according to the defined sequence identity threshold, and use the greedy algorithm to further equally split it based on complex type (i.e., the same WT complex will enter the same set)
        # 'loose': get the val_test set according to the defined sequence identity threshold, and equally split it into val and test sets randomly (i.e, the same WT complex may enter the same val and test sets).
        # 'noval': get the val_test set according to the defined sequence identity threshold, and treat the val_test set as the only test set directly
        criteria4identity = 'noval'
        # provide the generated sequence identity matrix path for current downstream set
        identity_path = 'D:/PROJECT B2_4/data/refer data/PPI/EquiPPI/data/{}_foldx_cleaned/{}_seq_identity.csv'.format(set_name, set_name)
        # threshold for splitting data based on sequence identity
        identity_threshold = 0.3
        # the path to store the generated data split file
        save_path = './data/{}_{}_data_split_{}.json'.format(set_name, split_mode, criteria4identity)

    downstream_pdbs = []
    with open(source_folder + f'{set_name}_chain_set{mutation_source}.jsonl') as f:
        counter = 0
        lines = f.readlines()
        for line in lines:
            counter += 1
            if counter % 2000 == 0:
                print('counter:', counter)
            entry = json.loads(line)
            name = entry['name']
            downstream_pdbs.append(name)

    downstream_pdbs = sorted(downstream_pdbs) # current pretraining_pdbs is sorted (containing all pdb in jsonl source data file)
    print(set_name, 'len(set(downstream_pdbs)):', len(set(downstream_pdbs)), 'len(downstream_pdbs):', len(downstream_pdbs))

    if split_mode == 'CV10_random':
        save_path = './data/{}_{}_data_split_{}.jsonl'.format(set_name, split_mode, random_seed)

        if not os.path.exists(save_path):
            with open(save_path, 'w') as f1:
                folds = 10
                all_folds = list(np.arange(folds))

                for i in range(folds):
                    val_fold = [i]
                    train_fold = sorted(list(set(all_folds) - set(val_fold)))
                    print('fold {}:'.format(i), train_fold, val_fold)

                    # pretraining_pdbs has been sorted above and the random seed has been fixed in data_split_for_training
                    # here pretraining_pdbs will be splited based on the sorted order
                    train_idx, val_idx = data_split_for_training(downstream_pdbs, folds, train_fold, val_fold)
                    print('train_idx, val_idx:', len(train_idx), len(val_idx))

                    train_set, val_set = (np.array(downstream_pdbs)[train_idx]).tolist(), (np.array(downstream_pdbs)[val_idx]).tolist()
                    downstream_data_split = {'train': train_set, 'val': val_set}
                    f1.write(json.dumps(downstream_data_split) + '\n')

        else:
            print('the required json data split has already been generated')

    elif split_mode == 'complex':
        # np.random.choice within this part of code should have used the defined random seed above
        save_path = './data/{}_{}_data_split_{}.jsonl'.format(set_name, split_mode, random_seed)

        if not os.path.exists(save_path):
            samplenum4complex = dict()
            for pdb in downstream_pdbs:
                pdb_ = pdb.split('__')[0] # widetype complex name
                if pdb_ not in samplenum4complex.keys():
                    samplenum4complex[pdb_] = 0
                    samplenum4complex[pdb_] += 1
                else:
                    samplenum4complex[pdb_] += 1
            # sort it from large to small (tuple format, (sample number, WT complex name))
            samplenum4complex = sorted(list(zip(samplenum4complex.values(), samplenum4complex.keys())), reverse=True)
            # print(samplenum4complex)

            batch_size = 5
            # create 5 bins based on sorted order
            a = [samplenum4complex[i:i + batch_size] for i in range(0, len(samplenum4complex), batch_size)]
            # transform each sample in a to a dict, key: WT complex name, value: sample number
            a = [[{i[1]: i[0]} for i in batch] for batch in a]
            # randomly select a complex for each fold from defined bins
            a = [np.random.choice(batch, len(batch), replace=False).tolist() for batch in a]
            # [[{'1R0R': 152}, {'3SGB': 190}, {'1JTG': 34}, {'1PPF': 171}, {'1A22': 58}], [...], [...], [...], [...]]

            with open(save_path, 'w') as f1:
                fold1, fold2, fold3, fold4, fold5 = [], [], [], [], []
                counter1 = 0
                for i in range(len(a)): # a: bins after the random complex selection for each fold
                    counter1 += 1
                    if counter1 < len(a): # current bin should be full (5 elements)
                        fold1.append(a[i][0])
                        fold2.append(a[i][1])
                        fold3.append(a[i][2])
                        fold4.append(a[i][3])
                        fold5.append(a[i][4])
                    elif counter1 == len(a): # current bin may not be full
                        # get the total allocated sample number for each fold until the last bin
                        fold1_complex_num = np.sum([i[list(i.keys())[0]] for i in fold1])
                        fold2_complex_num = np.sum([i[list(i.keys())[0]] for i in fold2])
                        fold3_complex_num = np.sum([i[list(i.keys())[0]] for i in fold3])
                        fold4_complex_num = np.sum([i[list(i.keys())[0]] for i in fold4])
                        fold5_complex_num = np.sum([i[list(i.keys())[0]] for i in fold5])
                        print('mutation sample numbers in each fold before the last bin:', fold1_complex_num, fold2_complex_num, fold3_complex_num, fold4_complex_num, fold5_complex_num)

                        temp_dict = {'fold1': fold1_complex_num, 'fold2': fold2_complex_num, 'fold3': fold3_complex_num, 'fold4': fold4_complex_num, 'fold5': fold5_complex_num}
                        samplenum4fold = sorted(list(zip(temp_dict.values(), temp_dict.keys())), reverse=False) # sort it from small to large
                        # print(samplenum4fold) # [(146, 'fold3'), (168, 'fold5'), (254, 'fold1'), (271, 'fold4'), (290, 'fold2')]

                        # sort the last bin from large to small, i[list(i.keys())[0]]: sample number, list(i.keys())[0]: WT complex name
                        temp_list = sorted([(i[list(i.keys())[0]], list(i.keys())[0]) for i in a[counter1-1]], reverse=True)
                        # print(temp_list) # [(1, '1CT0'), (1, '1CSO')]

                        # assign the last bin to the fold with fewer samples
                        # temp_list: from large to small, samplenum4fold: from small to large
                        for i in range(len(temp_list)):
                            num, complex_name = temp_list[i]
                            locals()[samplenum4fold[i][1]].append({complex_name: num}) # locals()[samplenum4fold[i][1]]: get variables for the specified folder
                        # to make WT complex number and the corresponding sample amount as equal as possible at the same time
                        print('total complex numbers in each fold:', len(fold1), len(fold2), len(fold3), len(fold4), len(fold5),
                              len(fold1) + len(fold2) + len(fold3) + len(fold4) + len(fold5), 'total complex number in current dataset:', len(samplenum4complex))

                # currently complexes for each fold has been determined
                # print(fold1) # [{'1R0R': 152}, {'1A4Y': 27}, {'1KTZ': 20}, {'1XD3': 12}]
                all_folds = list(np.arange(batch_size)) # 0, 1, 2, 3, 4
                for i in range(batch_size):
                    val_fold_index = [i]
                    train_fold_index = sorted(list(set(all_folds) - set(val_fold_index)))
                    # only get WT complex name from fold['N']
                    val_fold = [list(i.keys())[0] for i in locals()['fold' + str(val_fold_index[0]+1)]]
                    train_fold = [] # WT complex names contained in training set
                    for fold_index in train_fold_index:
                        train_fold.extend([list(i.keys())[0] for i in locals()['fold' + str(fold_index+1)]])
                    print('fold {}:'.format(i+1), train_fold_index, val_fold_index)

                    train_set, val_set = [], []
                    for pdb in downstream_pdbs:
                        pdb_ = pdb.split('__')[0]
                        if pdb_ in train_fold:
                            train_set.append(pdb)
                        elif pdb_ in val_fold:
                            val_set.append(pdb)
                        else:
                            print('outlier WT complex name:', pdb)
                    train_set, val_set = sorted(train_set), sorted(val_set)
                    downstream_data_split = {'train': train_set, 'val': val_set}
                    f1.write(json.dumps(downstream_data_split) + '\n')
                    print('total complex number in current training and validation set:', len(train_set), len(val_set), len(train_set)+len(val_set), 'total WT complex number:', len(downstream_pdbs))

        else:
            print('the required json data split has already been generated')

    elif split_mode == 'identity' and criteria4identity == 'strict':
        # get WT complex name for each set based on sequence identity
        train_name, val_test_name = get_proteinsplit_for_evaluation(identity_path, identity_threshold)

        # get samples corresponding to training set
        val_test_set = dict()
        train_set = []
        for pdb in downstream_pdbs:
            pdb_ = pdb.split('__')[0]
            if pdb_ in train_name: # pure widetype protein prefix
                train_set.append(pdb)
            if pdb_ in val_test_name: # protein types for validation and test
                if pdb_ not in val_test_set.keys():
                    val_test_set[pdb_] = 0
                    val_test_set[pdb_] += 1
                else:
                    val_test_set[pdb_] += 1
        train_set = sorted(train_set)

        # get samples corresponding to val and test sets
        val_test_set_ = sorted(list(zip(val_test_set.values(), val_test_set.keys())), reverse=True) # from large to small

        batch_size = 2
        a = [val_test_set_[i:i + batch_size] for i in range(0, len(val_test_set_), batch_size)]
        indicator = len(a[-1]) % 2 # check the element number in the last batch

        val_protein, test_protein = [], [] # the protein types involved in val and test sets
        if indicator: # odd
            sample = np.random.randint(0, 2, len(a) - 1)
        else: # even
            sample = np.random.randint(0, 2, len(a))
        # sample is the regulation for getting each part of data in a (how to get data from every batch in a)

        sample_ = sample.astype(bool)  # sample should be numpy format
        sample_ = ~sample_
        sample_ = sample_.astype(np.int32)

        for i in range(len(sample)):
            val_protein.append(a[i][sample[i]])
            test_protein.append(a[i][sample_[i]])

        if indicator:
            if sum(list(zip(*val_protein))[0]) > sum(list(zip(*test_protein))[0]): # unzip val_protein and test_protein, [(22, 16, 7, 5), ('1KTZ', '1AK4', '3K2M', '1YY9')]
                test_protein.extend(a[-1])
            else:
                val_protein.extend(a[-1])

        # val_protein: [(22, '1KTZ'), (16, '1AK4'), (7, '3K2M'), (5, '1YY9')]
        # test_protein: [(22, 'HM_1KTZ'), (16, 'HM_1YY9'), (9, '1FFW'), (5, '1JTG')]
        val_protein = list(zip(*val_protein))
        test_protein = list(zip(*test_protein))
        val_set, test_set = [], []
        for pdb in downstream_pdbs:
            pdb_ = pdb.split('__')[0]
            if pdb_ in set(val_protein[1]): # [(22, 16, 7, 5), ('1KTZ', '1AK4', '3K2M', '1YY9')]
                val_set.append(pdb)
            elif pdb_ in set(test_protein[1]):
                test_set.append(pdb)
            else:
                print('outlier pdb name:', pdb)
        val_set, test_set = sorted(val_set), sorted(test_set)
        # len(train_set), len(val_set), len(test_set): 521 50 52 check: 50 52 623
        print('len(train_set), len(val_set), len(test_set):', len(train_set), len(val_set), len(test_set), 'check:', len(train_set) + len(val_set) + len(test_set), sum(val_protein[0]), sum(test_protein[0]))
        downstream_data_split = {'train': train_set, 'val': val_set, 'test': test_set}

        if not os.path.exists(save_path):
            with open(save_path, 'w') as outfile:
                json.dump(downstream_data_split, outfile)
        else:
            print('the required json data split has already been generated')

    elif split_mode == 'identity' and criteria4identity == 'loose':
        # get WT complex name for each set based on sequence identity
        train_name, val_test_name = get_proteinsplit_for_evaluation(identity_path, identity_threshold)

        # get samples corresponding to each set
        train_set = []
        val_test_set = []
        for pdb in downstream_pdbs:
            pdb_ = pdb.split('__')[0]
            if pdb_ in train_name: # pure widetype protein prefix
                train_set.append(pdb)
            elif pdb_ in val_test_name:
                val_test_set.append(pdb)
            else:
                print('outlier pdb name:', pdb)
        train_set = sorted(train_set)
        val_test_set = sorted(val_test_set)

        folds = 2
        val_fold = [0]
        test_fold = [1]
        val_idx, test_idx = data_split_for_training(val_test_set, folds, val_fold, test_fold)
        val_set, test_set = (np.array(val_test_set)[val_idx]).tolist(), (np.array(val_test_set)[test_idx]).tolist()

        print('len(train_set), len(val_set), len(test_set):', len(train_set), len(val_set), len(test_set), 'check:', len(train_set) + len(val_set) + len(test_set))
        downstream_data_split = {'train': train_set, 'val': val_set, 'test': test_set}

        if not os.path.exists(save_path):
            with open(save_path, 'w') as outfile:
                json.dump(downstream_data_split, outfile)
        else:
            print('the required json data split has already been generated')

    elif split_mode == 'identity' and criteria4identity == 'noval':
        # get WT complex name for each set based on sequence identity
        train_name, val_test_name = get_proteinsplit_for_evaluation(identity_path, identity_threshold)

        # get samples corresponding to each set
        train_set = []
        val_test_set = []
        for pdb in downstream_pdbs: # current pretraining_pdbs is sorted (containing all pdb in jsonl source data file)
            pdb_ = pdb.split('__')[0]
            if pdb_ in train_name: # pure widetype protein prefix
                train_set.append(pdb)
            elif pdb_ in val_test_name:
                val_test_set.append(pdb)
            else:
                print('outlier pdb name:', pdb)

        train_set = sorted(train_set)
        val_test_set = sorted(val_test_set)

        print('len(train_set), len(val_test_set):', len(train_set), len(val_test_set), 'check:', len(train_set) + len(val_test_set))
        downstream_data_split = {'train': train_set, 'val': val_test_set}

        if not os.path.exists(save_path):
            with open(save_path, 'w') as outfile:
                json.dump(downstream_data_split, outfile)
        else:
            print('the required json data split has already been generated')

    elif split_mode == 'plotting': # for drawing t-SNE plotting for the whole downstream dataset
        save_path = './data/{}_{}_data_split.json'.format(set_name, split_mode)
        downstream_data_split = {'train': downstream_pdbs, 'val': [], 'test': []}

        with open(save_path, 'w') as outfile:
            json.dump(downstream_data_split, outfile)

    else:
        print('current data splitting mode is not supported: {}'.format(split_mode))






















