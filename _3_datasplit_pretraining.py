# step3:
# data split for pretraining set (training-validation)
import random
import json
import numpy as np


# 1, 429, 1002, 1012, 1024
random_seed = 1024
random.seed(random_seed)
np.random.seed(random_seed)
print('random_seed:', random_seed)

folds = 10
train_fold = [0, 1, 2, 3, 4, 5, 6, 7, 8]
val_fold = [9] # 10% as the validation set

def data_split_for_training(all_pretraining_index, folds, train_fold, val_fold):

    prng = np.random.RandomState(random_seed)
    allindex = prng.permutation(len(all_pretraining_index))
    pos_inter_fold = np.array_split(allindex, folds)

    train_pos_sample = [pos_inter_fold[i] for i in train_fold]
    val_pos_sample = [pos_inter_fold[i] for i in val_fold]

    # train_pos_sample: array type
    train_pos_sample = np.concatenate(train_pos_sample)
    val_pos_sample = np.concatenate(val_pos_sample)

    # print('train_pos_sample:', len(train_pos_sample), train_pos_sample)
    # print('val_pos_sample', len(val_pos_sample), val_pos_sample)

    # sorted: list type
    return np.array(sorted(train_pos_sample)), np.array(sorted(val_pos_sample))


if __name__ == '__main__':
    # the path to read the json source data file produced by _3_generate_json_input.py
    source_folder = './data/refer data/PPI/EquiPPI/data/'
    # the path to store the generated data split file
    save_path = './data/refer data/PPI/EquiPPI/data/pretraining_data_split.json'

    pretraining_pdbs = []
    with open(source_folder + 'pretraining_chain_set.jsonl') as f:
        counter = 0
        lines = f.readlines()
        for line in lines:
            counter += 1
            if counter % 2000 == 0:
                print('counter:', counter)
            entry = json.loads(line)
            name = entry['name']
            pretraining_pdbs.append(name)

    # pretraining_pdbs = [i[:-4] for i in os.listdir(source_folder + 'pretraining_set_foldx/')]
    print('len(set(pretraining_pdbs)):', len(set(pretraining_pdbs)), 15219) # original one: 15219

    pretraining_pdbs = sorted(pretraining_pdbs)
    print('len(pretraining_pdbs):', len(pretraining_pdbs), 15219) # current pretraining_pdbs is sorted

    # pretraining_pdbs has been sorted above
    # here pretraining_pdbs will be splited based on the sorted order
    train_idx, val_idx = data_split_for_training(pretraining_pdbs, folds, train_fold, val_fold)
    print('train_idx, val_idx:', len(train_idx), len(val_idx))

    train_set, val_set = (np.array(pretraining_pdbs)[train_idx]).tolist(), (np.array(pretraining_pdbs)[val_idx]).tolist()
    pretraining_data_split = {'train': train_set, 'val': val_set}

    with open(save_path, 'w') as outfile:
        json.dump(pretraining_data_split, outfile)

