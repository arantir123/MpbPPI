# the script for running our pre-trained MpbPPI for ddg prediction (based on GBT selected downstream features)
import time
import pandas as pd
import torch
import gvp.models
import tqdm
import numpy as np
import json
import torch_geometric
from functools import partial
import random
from config import ap
import os
from torch_geometric.nn.pool import global_max_pool, global_mean_pool
from _3_generate_residuefeats_finetuning import FinetuningCVDataset, FinetuningGraphDataset, BatchSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import scipy.stats
print = partial(print, flush=True)

# fix random seed
random_seed = 1234
random.seed(random_seed) # mainly for controlling BatchSampler to have the same batch organization every time we run the whole code
np.random.seed(random_seed) # in PretrainingGraphDataset, the random noise and mask are generated based on np.random
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print('random_seed:', random_seed)

# seed for retrieving corresponding downstream dataset splitting
splitting_seed = 256
# indicate the MT PPI complex source for retrieving corresponding data source file
mutation_source = '_foldx'
# indicate whether to add mutation cr_token into the interface cr_token set (only set to True for M1101)
# for M1101 to solve the cases that interface cannot be retrieved by pymol
add_mut_to_interface = False
# intermediate feat dim in gvp
node_dim = (256, 16)
edge_dim = (32, 1)
device = "cuda" if torch.cuda.is_available() else "cpu"

# overall hyper-parameters for retrieving corresponding pretraining models, to avoid hyper-parameter conflict between pretraining and downstream calculation
pretraining_lr = 0.001
pretraining_epochs = 100
pretraining_early_stop = 30

dataloader = lambda x: torch_geometric.data.DataLoader(x, # x: pytorch geometric Data
                        num_workers=args.num_workers,
                        pin_memory=False,
                        # defines the strategy to draw samples from the dataset
                        # BatchSampler: set a maximum residue number in each batch, and based on this number to put different resides into different batches
                        # in BatchSampler, shuffle=True，thus every time to call this dataloader, the generated batches could be different (rather than indicating shuffling batches before each epoch) if random seed is not fixed
                        # if set randon with a fixed random seed, the generated batches are the same every time we run this code
                        batch_sampler=BatchSampler(
                            # max number of nodes per batch
                            x.node_counts, max_nodes=args.max_nodes))


def main(args):
    # for reading pretraining model specified by these args names
    model_save_pretraining = 'MpbPPI_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(pretraining_lr, pretraining_epochs, pretraining_early_stop, args.noise_type, args.noise_prob, args.whether_SASA,
        args.ca_denoising_weight, args.sidec_denoising_weight, args.sasa_pred_weight, args.AA_prediction_weight, args.main_num_layers, args.aux_layer_list, args.whether_sidec_noise, args.aux_med_dropout, args.only_CA, args.top_k, node_dim + edge_dim,
        args.whether_AA_prediction, args.whether_sidec_prediction, args.sidec_chain_normalization, args.whether_spatial_graph, args.graph_cat)
    # sklearn GBT model hyper-parameters
    model_save_finetuning = 'MpbPPIdecoder_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.set_name, args.learning_rate, args.difference_feats, args.global_feats, node_dim + edge_dim,
        args.max_depth, args.max_features, args.min_samples_split, args.n_estimators, args.subsample, args.n_iter_no_change, args.validation_fraction)
    print('model_save_pretraining:', model_save_pretraining)
    print('model_save_finetuning:', model_save_finetuning)

    if not os.path.exists('storage_finetuning/'):
        os.makedirs('storage_finetuning/')

    if args.data_split_mode == 'identity' and args.data_split_tag == '_noval':
        with open(args.data_dir + f'{args.set_name}_{args.data_split_mode}_data_split{args.data_split_tag}.json') as f:
            split_list = [json.load(f)]
    elif args.data_split_mode == 'CV10_random' or args.data_split_mode == 'complex':
        with open(args.data_dir + f'{args.set_name}_{args.data_split_mode}_data_split{args.data_split_tag}_{splitting_seed}.jsonl') as f:
            splits = f.readlines()
        split_list = []
        for split in splits:
            split_list.append(json.loads(split))
    else:
        print('current script does not support current data splitting')
        raise NotImplementedError

    MSE_total, RMSE_total, MAE_total, PEARSON_total, label_total, prediction_total, name_total = [], [], [], [], [], [], []
    MSE_featselect_total, RMSE_featselect_total, MAE_featselect_total, PEARSON_featselect_total = [], [], [], []
    for fold in range(len(split_list)):
        print('current fold:', fold)
        if os.path.exists('./storage_pretraining/' + model_save_pretraining):
            model = gvp.models.MR_EquiPPIModel((29, 9), node_dim, (32, 1), edge_dim, num_layers=args.main_num_layers, drop_rate=args.main_dropout, graph_cat = args.graph_cat).to(device)

            GBT_params = {'random_state': random_seed, 'learning_rate': args.learning_rate, 'max_depth': args.max_depth, 'max_features': args.max_features,
                          'min_samples_split': args.min_samples_split, 'n_estimators': args.n_estimators, 'subsample':args.subsample, 'n_iter_no_change':args.n_iter_no_change, 'validation_fraction':args.validation_fraction}
            decoder = GradientBoostingRegressor(**GBT_params)

            print('Loading downstream dataset ...')
            downstream_set = FinetuningCVDataset(path=args.data_dir + f'{args.set_name}_chain_set{mutation_source}.jsonl', dataset_splits=split_list[fold])

            # train_set, val_set = map(FinetuningGraphDataset, (downstream_set.train, downstream_set.val))
            train_set = FinetuningGraphDataset(downstream_set.train, sidec_chain_normalization=args.sidec_chain_normalization, whether_spatial_graph=args.whether_spatial_graph, add_mut_to_interface=add_mut_to_interface)
            val_set = FinetuningGraphDataset(downstream_set.val, sidec_chain_normalization=args.sidec_chain_normalization, whether_spatial_graph=args.whether_spatial_graph, add_mut_to_interface=add_mut_to_interface)

            # main training & evaluation process
            MSE_test, RMSE_test, MAE_test, PEARSON_test, MSE_featselect_test, RMSE_featselect_test, MAE_featselect_test, PEARSON_featselect_test, label_test, prediction_test, name_test = \
                finetuning(model_save_pretraining, model_save_finetuning, model, decoder, train_set, val_set, args)

            MSE_total.append(MSE_test)
            RMSE_total.append(RMSE_test)
            MAE_total.append(MAE_test)
            PEARSON_total.append(PEARSON_test)

            MSE_featselect_total.append(MSE_featselect_test)
            RMSE_featselect_total.append(RMSE_featselect_test)
            MAE_featselect_total.append(MAE_featselect_test)
            PEARSON_featselect_total.append(PEARSON_featselect_test)

            name_total.append(name_test)
            prediction_total.append(prediction_test)
            label_total.append(label_test)

        else:
            print('the specified pretraining model cannot be found in: {},'.format('./storage_pretraining/' + model_save_pretraining), 'fail to finetune the model')

    # end of the CV loop
    for name, value in vars(args).items():
        print(name, value)
    print('node_dim:', node_dim, 'edge_dim:', edge_dim, 'splitting_seed:', splitting_seed, 'mutation_source:', mutation_source, 'add_mut_to_interface:', add_mut_to_interface)
    print('*** Above Are All Hyper Parameters ***')

    # print overall evaluation results
    print('average MSE, RMSE, MAE, Pearson on test set:', np.mean(MSE_total), np.mean(RMSE_total), np.mean(MAE_total), np.mean(PEARSON_total))
    print('average MSE, RMSE, MAE, Pearson on test set after GBT feature selection:', np.mean(MSE_featselect_total), np.mean(RMSE_featselect_total), np.mean(MAE_featselect_total), np.mean(PEARSON_featselect_total))

    # output overall prediction results (based on GBT selected features)
    pd_columns = ['Name_total', 'Label_total', 'Prediction_total']
    name_total, label_total, prediction_total = np.concatenate(name_total).reshape(-1, 1), np.concatenate(label_total).reshape(-1, 1), np.concatenate(prediction_total).reshape(-1, 1)
    save_file = pd.DataFrame(np.concatenate([name_total, label_total, prediction_total], axis=1), columns=pd_columns)
    save_file.to_csv('./storage_finetuning/featselect_crossvalidation_{}_{}{}.csv'.format(args.set_name, args.data_split_mode, mutation_source))

    print('model_save_pretraining:', model_save_pretraining)
    print('model_save_finetuning:', model_save_finetuning)


def finetuning(model_save_pretraining, model_save_finetuning, model, decoder, train_set, val_set, args):
    train_loader, val_loader = map(dataloader, (train_set, val_set))

    # load model
    checkpoint = torch.load('./storage_pretraining/' + model_save_pretraining)
    model.load_state_dict(checkpoint['EquiPPI'])
    model.to(device)
    model.eval()
    t0 = time.time()

    # training, becuase in every fold, samples in training and valiation sets are different, thus in different folds, the batch/dataloader organization ways/orders are different
    with torch.no_grad():
        train_X, train_Y, train_name = loop(model, train_loader, args)
    train_X = np.round(train_X, 3)

    print('start the GBT decoder training ...')
    decoder.fit(train_X, train_Y)

    # test
    with torch.no_grad():
        # test_name is organized as np.array format
        test_X, test_Y, test_name = loop(model, val_loader, args)
    test_X = np.round(test_X, 3)

    # test_prediction = np.clip(decoder.predict(test_X), -8.0, 8.0)
    test_prediction = decoder.predict(test_X)

    MSE_test = mean_squared_error(test_Y, test_prediction)
    RMSE_test = np.sqrt(MSE_test)
    MAE_test = mean_absolute_error(test_Y, test_prediction)
    PEARSON_test = scipy.stats.pearsonr(test_Y.reshape(-1), test_prediction.reshape(-1))[0]

    t1 = time.time()
    print(f'total elapsed time of normal training and testing in current fold: {t1 - t0:.4f}')
    print(f'normal evaluation metrics in current fold, MSE: {MSE_test:.4f}, RMSE: {RMSE_test:.4f}, MAE: {MAE_test:.4f}, Pearson: {PEARSON_test:.4f}')

    # evaluate the model performance based on pre-defined number of features (screened by GBT feature importance)
    # https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
    decoder_feature_importance = decoder.feature_importances_
    sorted_idx = np.argsort(decoder_feature_importance)[::-1] # the importance is sorted from large to small
    decoder.fit(train_X[:, sorted_idx[:args.GBT_maxfeats_num]], train_Y)
    test_featselect_prediction = decoder.predict(test_X[:, sorted_idx[:args.GBT_maxfeats_num]])

    MSE_featselect_test = mean_squared_error(test_Y, test_featselect_prediction)
    RMSE_featselect_test = np.sqrt(MSE_featselect_test)
    MAE_featselect_test = mean_absolute_error(test_Y, test_featselect_prediction)
    PEARSON_featselect_test = scipy.stats.pearsonr(test_Y.reshape(-1), test_featselect_prediction.reshape(-1))[0]

    t2 = time.time()
    print(f'total elapsed time of GBT feature selection based testing in current fold: {t2 - t1:.4f}')
    print(f'evaluation metrics after GBT feature selection in current fold, MSE: {MSE_featselect_test:.4f}, RMSE: {RMSE_featselect_test:.4f}, MAE: {MAE_featselect_test:.4f}, Pearson: {PEARSON_featselect_test:.4f}')

    # output the results based on GBT feature selection
    return MSE_test, RMSE_test, MAE_test, PEARSON_test, MSE_featselect_test, RMSE_featselect_test, MAE_featselect_test, PEARSON_featselect_test, test_Y, test_featselect_prediction, test_name


def loop(model, dataloader, args=None):
    t = tqdm.tqdm(dataloader, ncols=75)

    encoder_embeddings = []
    ddg_label_list = []
    name_list = []
    for batch in t:
        batch = batch.to(device)
        # gvp encoder input
        wt_h_V = (batch.wt_node_s, batch.wt_node_v)
        wt_h_E = (batch.wt_edge_s, batch.wt_edge_v)
        mt_h_V = (batch.mt_node_s, batch.mt_node_v)
        mt_h_E = (batch.mt_edge_s, batch.mt_edge_v)

        wt_extra_h_E = (batch.wt_extra_edge_s, batch.wt_extra_edge_v)
        mt_extra_h_E = (batch.mt_extra_edge_s, batch.mt_extra_edge_v)
        wt_encoder_embeddings = model(wt_h_V, batch.wt_edge_index, wt_h_E, batch.wt_extra_edge_index, wt_extra_h_E)
        mt_encoder_embeddings = model(mt_h_V, batch.mt_edge_index, mt_h_E, batch.mt_extra_edge_index, mt_extra_h_E)
        # print(wt_encoder_embeddings.size(), mt_encoder_embeddings.size()) # torch.Size([2525, 148]) torch.Size([2525, 148])

        # final feature generation
        mutation_mask = batch.mutation_mask
        interface_mask = batch.interface_mask
        mask_size = batch.mask_size # tensor([615, 619, 676, 615], device='cuda:0')
        graph_id_batch = []
        for i in torch.arange(mask_size.size(0)):
            graph_id_batch.append(i.expand(mask_size[i]))
        graph_id_batch = torch.cat(graph_id_batch).to(device) # get the residue allocation for current batch

        # the residue number check between WT and MT has been conducted in pytorch Dataset
        wt_mutation_site_max = global_max_pool(x=wt_encoder_embeddings[mutation_mask], batch=graph_id_batch[mutation_mask], size=mask_size.size(0))
        wt_mutation_site_mean = global_mean_pool(x=wt_encoder_embeddings[mutation_mask], batch=graph_id_batch[mutation_mask], size=mask_size.size(0))
        wt_interface_site_max = global_max_pool(x=wt_encoder_embeddings[interface_mask], batch=graph_id_batch[interface_mask], size=mask_size.size(0))
        wt_interface_site_mean = global_mean_pool(x=wt_encoder_embeddings[interface_mask], batch=graph_id_batch[interface_mask], size=mask_size.size(0))
        # print(wt_mutation_site_max, torch.max(wt_encoder_embeddings[615: 615+619][mutation_mask[615: 615+619]], 0)[0]) # the corresponding embeddings should be the same
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_max_pool.html

        mt_mutation_site_max = global_max_pool(x=mt_encoder_embeddings[mutation_mask], batch=graph_id_batch[mutation_mask], size=mask_size.size(0))
        mt_mutation_site_mean = global_mean_pool(x=mt_encoder_embeddings[mutation_mask], batch=graph_id_batch[mutation_mask], size=mask_size.size(0))
        mt_interface_site_max = global_max_pool(x=mt_encoder_embeddings[interface_mask], batch=graph_id_batch[interface_mask], size=mask_size.size(0))
        mt_interface_site_mean = global_mean_pool(x=mt_encoder_embeddings[interface_mask], batch=graph_id_batch[interface_mask], size=mask_size.size(0))
        intergrate_embedding = [wt_mutation_site_max, wt_mutation_site_mean, wt_interface_site_max, wt_interface_site_mean,
                                mt_mutation_site_max, mt_mutation_site_mean, mt_interface_site_max, mt_interface_site_mean]

        if args.difference_feats: # WT/MT mutation site difference features
            intergrate_embedding = torch.cat(intergrate_embedding + [wt_mutation_site_max-mt_mutation_site_max, wt_mutation_site_mean-mt_mutation_site_mean], 1)
        else:
            intergrate_embedding = torch.cat(intergrate_embedding, 1)

        if args.global_feats: # global information of all residues in a protein
            intergrate_embedding = torch.cat([intergrate_embedding, global_mean_pool(x=mt_encoder_embeddings, batch=graph_id_batch, size=mask_size.size(0))], 1) # torch.Size([4, 1628])

        output_detach = intergrate_embedding.cpu().detach().numpy()
        encoder_embeddings.append(output_detach)

        ddg_label = batch.ddg
        ddg_label_detach = ddg_label.cpu().detach().numpy()
        ddg_label_list.append(ddg_label_detach)
        name_list.append(batch.name)

        torch.cuda.empty_cache()

    encoder_embeddings = np.concatenate(encoder_embeddings)
    ddg_label_list = np.concatenate(ddg_label_list)
    name_list = np.concatenate(name_list)

    return encoder_embeddings, ddg_label_list, name_list


if __name__ == '__main__':
    args = ap.parse_args()

    # *** downstream setting related hyperparameters ***
    # the root path to store data source files
    args.data_dir = 'data/'
    # the specified downstream ddg dataset names
    args.set_name = 'S4169'
    # data_split_mode: 'CV10_random'/'complex'
    args.data_split_mode = 'CV10_random'
    # extra tag for specifying required data splitting files (default: '')
    args.data_split_tag = ''
    args.num_workers = 4 # 4

    # GBT related hyperparameters
    args.learning_rate = 0.001
    args.subsample = 0.7 # 0.3/0.4/0.7
    args.min_samples_split = 3
    args.max_depth = 6 # 4/6/8
    args.max_features = 'sqrt'
    args.n_estimators = 50000 # 30000/40000/50000
    # n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving,
    # by default it is set to None to disable early stopping (otherwise it is an integar in the range [1, inf))
    args.n_iter_no_change = None
    # validation_fraction must be in the range (0.0, 1.0), only used if n_iter_no_change is set to an integer
    args.validation_fraction = 0.1
    # whether to add difference information between generated WT and MT complexes as an extra feature
    args.difference_feats = True
    # whether to add global information of all residues in a protein as an extra feature
    args.global_feats = True

    # * extra hyper-parameter to select more important features during downstream predictions *
    args.GBT_maxfeats_num = 2500

    # *** pretraining related hyperparameters (for retrieving the corresponding pretrained MpbPPI models) ***
    # *** Note: there are some conflicting hyperparameters between pretraining and finetuning shown before the main function ***
    args.main_num_layers = 5
    # K value for the KNN graph of each protein
    args.top_k = 20
    # noise type to be added into original coordinates, choice={trunc_normal, normal, uniform}
    args.noise_type = 'trunc_normal'
    # the probability for adding noise to each residue of a protein
    args.noise_prob = 0.15
    # whether only to add noise to CA rather than all backbone atoms
    args.only_CA = False
    # whether to use side chain prediction as a pretraining task
    args.whether_sidec_prediction = True
    # whether to add the same type noise as CA coordinates to side chain atoms
    args.whether_sidec_noise = True
    args.sidec_chain_normalization = True
    # whether to use SASA prediction as an auxiliary task in pretraining
    args.whether_SASA = True
    args.whether_AA_prediction = True
    args.whether_spatial_graph = True
    # loss weight ratio for CA coordinate denoising task
    args.ca_denoising_weight = 1
    # loss weight ratio for side chain coordinate information denoising task
    args.sidec_denoising_weight = 1
    # loss weight ratio for SASA prediction task
    args.sasa_pred_weight = 1
    # loss weight ratio for AA prediction task
    args.AA_prediction_weight = 1
    # neuron unit number list (except for the input unit number) for the multi-task pretraining predictors
    args.aux_layer_list = [512, 128, 3]
    # the intermediate layer dropout rate of the multi-task pretraining predictors
    args.aux_med_dropout = 0.2
    args.graph_cat = 'cat'

    for name,value in vars(args).items():
            print(name,value)
    print('node_dim:', node_dim, 'edge_dim:', edge_dim, 'splitting_seed:', splitting_seed, 'mutation_source:', mutation_source, 'add_mut_to_interface:', add_mut_to_interface)
    print('*** Above Are All Hyper Parameters ***')

    main(args)