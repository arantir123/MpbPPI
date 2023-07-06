# step4:
# construct the GEE and pre-train it with the multi-task strategy (the pre-training process includes AA denoising + adding radius graph)
import torch
import gvp.models
import tqdm
import numpy as np
import pandas as pd
import torch_geometric
from functools import partial
import random
import torch.optim as opt
import itertools
import time
import copy
from config import ap
from _3_generate_residuefeats_pretraining import PretrainingDataset, PretrainingGraphDataset, BatchSampler
from gvp.models import AutomaticWeightedLoss
print = partial(print, flush=True)

# fix random seed
random_seed = 1234
# check whether they are related to random noise generation
random.seed(random_seed) # mainly for controlling BatchSampler to have the same batch organization every time we run the whole code
np.random.seed(random_seed) # in PretrainingGraphDataset, the random noise and mask are generated based on np.random
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print('random_seed:', random_seed)

# original input feature dim
# node_s=[126, 29], node_v=[126, 9, 3], edge_s=[3780, 32], edge_v=[3780, 1, 3]
# intermediate feature dimension to be tuned in GEE
node_dim = (256, 16)
edge_dim = (32, 1)
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    # name for model storage_pretraining, currently we use the scheme that overwrites the previous model with the existing best model based on validation set
    model_save_name = 'MpbPPI_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt'.format(args.lr, args.epochs, args.early_stop, args.noise_type, args.noise_prob, args.whether_SASA,
        args.ca_denoising_weight, args.sidec_denoising_weight, args.sasa_pred_weight, args.AA_prediction_weight, args.main_num_layers, args.aux_layer_list, args.whether_sidec_noise, args.aux_med_dropout, args.only_CA, args.top_k, node_dim + edge_dim,
        args.whether_AA_prediction, args.sidec_chain_normalization, args.whether_spatial_graph, args.graph_cat, args.whether_sidec_prediction)
    # a different between GVP-GNN and EGNN is that GVP-GNN can process multiple groups of vector features but original GVP-GNN cannot gvp the intersection between s and V
    # num_layers of GVP-GNN: 3-6, default GVP number in one GVP-GNN for processing node and edge features is 3 (consider fix it at first)
    model = gvp.models.MR_EquiPPIModel((29, 9), node_dim, (32, 1), edge_dim, num_layers=args.main_num_layers, drop_rate=args.main_dropout, graph_cat = args.graph_cat).to(device) # graph_cat = 'sum'/'mean'/'cat
    predictor = gvp.models.multi_task_predictor(args.whether_SASA, node_dim[0] + node_dim[1] * 3, args.aux_layer_list, args.aux_in_dropout, args.aux_med_dropout, args.whether_AA_prediction, args.whether_sidec_prediction).to(device)
    print('Loading pretraining dataset')

    pretraining_set = PretrainingDataset(path=args.data_dir + 'pretraining_chain_set.jsonl', splits_path=args.data_dir + 'pretraining_data_split.json')

    # hyperparameters: data_list, noise_type: str, noise: float = 1.0, mask_prob: float = 0.15,
    # only_CA = True (only denoising CA in backbone atoms), if_sidec_noise = True (whether to add similar scale noise to side chain atoms), SASA_mask = True (whether consider SASA as a pretraining objective),
    # num_positional_embeddings=16, top_k = 30, num_rbf=16, device='cpu'
    print('\nNumbers of complex samples in training and validation sets:', len(pretraining_set.train), len(pretraining_set.val))

    train_set = PretrainingGraphDataset(pretraining_set.train, args.noise_type, 1.0, args.noise_prob, only_CA=args.only_CA, if_sidec_noise=args.whether_sidec_noise,
                                        SASA_mask=args.whether_SASA, top_k=args.top_k, sidec_chain_normalization=args.sidec_chain_normalization, whether_AA_prediction=args.whether_AA_prediction,
                                        whether_spatial_graph=args.whether_spatial_graph, whether_sidec_prediction=args.whether_sidec_prediction)
    val_set = PretrainingGraphDataset(pretraining_set.val, args.noise_type, 1.0, args.noise_prob, only_CA=args.only_CA, if_sidec_noise=args.whether_sidec_noise,
                                      SASA_mask=args.whether_SASA, top_k=args.top_k, sidec_chain_normalization=args.sidec_chain_normalization, whether_AA_prediction=args.whether_AA_prediction,
                                      whether_spatial_graph=args.whether_spatial_graph, whether_sidec_prediction=args.whether_sidec_prediction)

    pretraining(model_save_name, model, predictor, train_set, val_set, args)

    for name, value in vars(args).items():
        print(name, value)
    print('node_dim:', node_dim, 'edge_dim:', edge_dim)
    print('*** Above Are All Hyper Parameters ***')
    print('model_save_name:', model_save_name)


def pretraining(model_save_name, model, predictor, train_set, val_set, args):
    train_loader, val_loader = map(dataloader, (train_set, val_set))

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=args.lr)
    # awl = AutomaticWeightedLoss(loss_num)
    # optimizer = torch.optim.Adam(
    #     [{'params': itertools.chain(model.parameters(), predictor.parameters())}, {'params': awl.parameters(), 'weight_decay': 0}], lr=args.lr)
    # Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    # This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.
    # new_lr = lr * factor
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=args.lr_scheduler_patience, min_lr=5e-6) # 0.000005

    # tot_params = sum([np.prod(p.size()) for p in model.parameters()]) # the parameters about inference in CPD task has been removed from EquiPPI
    # print(f"Total number of parameters in EquiPPIModel: {tot_params}")
    # tot_params = sum([np.prod(p.size()) for p in predictor.parameters()])
    # print(f"Total number of parameters in pretraining predictors: {tot_params}")

    t0 = time.time()
    val_loss_list = []
    best_loss, best_epoch, bad_cnt, counter1 = np.inf, 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        counter1 += 1
        t1 = time.time()
        model.train()
        predictor.train()

        # loop represents batch-wise training/evaluation for current epoch
        loss = loop(model, predictor, train_loader, optimizer=optimizer) # loss returned is the average loss of current epoch with float type
        print(f'EPOCH {epoch} TRAIN loss: {loss:.4f}, best_loss: {best_loss:.4f}, best_epoch: {best_epoch}')

        model.eval()
        predictor.eval()
        with torch.no_grad():
            if not epoch % args.valid_steps:
                loss = loop(model, predictor, val_loader) # no optimizer is sent to loop here becuase it is no need to update model parameters here
                val_loss_list.append(loss)

                if loss < best_loss: # use overall loss as model improvement criteria
                    best_loss = loss
                    best_epoch = epoch
                    bad_cnt = 0 # patience counter
                    best_model = copy.deepcopy({
                        'EquiPPI': model.state_dict(),
                        'predictor': predictor.state_dict()})
                else:
                    bad_cnt += 1
                    if bad_cnt == args.early_stop:
                        print('early stopping, current epoch number: {}'.format(epoch))
                        break # finish epoch loops

                print(f'EPOCH {epoch} VAL loss: {loss:.4f}, best_loss: {best_loss:.4f}, best_epoch: {best_epoch}')
                # metrics: the metrics you want to use to measure whether there is performance change (usually from the validation set)
                # in every complete epoch, lr_scheduler can be called after training process (model.train()) and validation process (model.eval() + with torch.no_grad())
                # and it is driven by 'metrics' generated from validation process
                metrics = loss
                # def step(self, metrics: Any, epoch: Optional[int]=...) -> None: ...
                lr_scheduler.step(metrics)

        print('learning rate in current epoch {}:'.format(epoch), lr_scheduler.optimizer.param_groups[0]['lr'])
        t2 = time.time()
        # if counter1 == 1:
            # print('approximate elapsed time for finishing one epoch:', t2 - t1)

    t3 = time.time()
    print('total elapsed time in all epochs:', t3 - t0)
    # save the best model
    torch.save(best_model, './storage_pretraining/' + model_save_name)
    # with torch.no_grad():
    #     checkpoint = torch.load('./storage_pretraining/' + model_save_name)
    #     model.load_state_dict(checkpoint['EquiPPI'])
    #     predictor.load_state_dict(checkpoint['predictor'])

    # save the validation loss curve
    pd.DataFrame(val_loss_list, columns=['VAL_LOSS']).to_csv('./storage_pretraining/' + model_save_name + '.csv')


def loop(model, predictor, dataloader, optimizer=None):
    t = tqdm.tqdm(dataloader, ncols=75)

    # all the pretraining tasks are regression task (SASA_label, sidec_label, X_ca_label), which can be measured by MSELoss
    sigmoid = torch.nn.Sigmoid()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss_fn2 = torch.nn.BCELoss(reduction='mean')
    total_loss, total_count = 0, 0 # record the corresponding value for current epoch

    # if use tqdm to output progress, if print occurs in the loop of tqdm, tqdm will be interrupted and generate multiple bars
    for batch in t:
        # in every batch, one time zero_grad() and one time backward() will be performed
        if optimizer: optimizer.zero_grad()
        batch = batch.to(device)

        # gvp encoder input
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        extra_h_E = (batch.extra_edge_s, batch.extra_edge_v)
        encoder_embeddings = model(h_V, batch.edge_index, h_E, batch.extra_edge_index, extra_h_E)[batch.mask] # get embeddings corresponding to randomly masked samples
        # print(encoder_embeddings.size()) # torch.Size([55, 148])

        # pretraining labels
        original_ca_label = batch.original_ca_label
        new_ca_label = batch.new_ca_label
        sidec_label = batch.sidec_label
        SASA_label = batch.SASA_label
        AA_prediction_label = batch.AA_prediction_label
        assert encoder_embeddings.size(0) == original_ca_label.size(0) == new_ca_label.size(0) == sidec_label.size(0) == SASA_label.size(0) == AA_prediction_label.size(0)

        if args.whether_SASA == True and args.whether_AA_prediction == True and args.whether_sidec_prediction == True:
            predicted_ca_coord, predicted_sidec_info, predicted_SASA, predicted_AA = predictor(encoder_embeddings, new_ca_label)
            ca_denoising_loss = loss_fn(predicted_ca_coord, original_ca_label)
            sidec_denoising_loss = loss_fn(predicted_sidec_info, sidec_label)
            SASA_prediction_loss = loss_fn(predicted_SASA, SASA_label)
            AA_prediction_loss = loss_fn2(sigmoid(predicted_AA), AA_prediction_label)
            # print('loss magnitude for current batch:', ca_denoising_loss, sidec_denoising_loss, SASA_prediction_loss, AA_prediction_loss)
            # loss magnitude for current batch: tensor(0.4261) tensor(94.0827) (around 20 times to the first one) tensor(0.1858) (ca_denoising_loss, sidec_denoising_loss, SASA_prediction_loss)
            # loss magnitude for current batch: tensor(0.9612) tensor(0.1915) tensor(0.5101) tensor(0.7329) (ca_denoising_loss, sidec_denoising_loss, SASA_prediction_loss, AA_prediction_loss)
            # looks the sidec_denoising_loss magnitude is larger than that of other losses
            total_loss_batch = ca_denoising_loss * args.ca_denoising_weight + sidec_denoising_loss * args.sidec_denoising_weight + SASA_prediction_loss * args.sasa_pred_weight + AA_prediction_loss * args.AA_prediction_weight

        elif args.whether_SASA == True and args.whether_AA_prediction == False and args.whether_sidec_prediction == True:
            predicted_ca_coord, predicted_sidec_info, predicted_SASA = predictor(encoder_embeddings, new_ca_label)
            ca_denoising_loss = loss_fn(predicted_ca_coord, original_ca_label)
            sidec_denoising_loss = loss_fn(predicted_sidec_info, sidec_label)
            SASA_prediction_loss = loss_fn(predicted_SASA, SASA_label)
            total_loss_batch = ca_denoising_loss * args.ca_denoising_weight + sidec_denoising_loss * args.sidec_denoising_weight + SASA_prediction_loss * args.sasa_pred_weight

        elif args.whether_SASA == False and args.whether_AA_prediction == True and args.whether_sidec_prediction == True:
            predicted_ca_coord, predicted_sidec_info, predicted_AA = predictor(encoder_embeddings, new_ca_label)
            ca_denoising_loss = loss_fn(predicted_ca_coord, original_ca_label)
            sidec_denoising_loss = loss_fn(predicted_sidec_info, sidec_label)
            AA_prediction_loss = loss_fn2(sigmoid(predicted_AA), AA_prediction_label)
            total_loss_batch = ca_denoising_loss * args.ca_denoising_weight + sidec_denoising_loss * args.sidec_denoising_weight + AA_prediction_loss * args.AA_prediction_weight

        elif args.whether_SASA == True and args.whether_AA_prediction == True and args.whether_sidec_prediction == False:
            predicted_ca_coord, predicted_SASA, predicted_AA = predictor(encoder_embeddings, new_ca_label)
            ca_denoising_loss = loss_fn(predicted_ca_coord, original_ca_label)
            SASA_prediction_loss = loss_fn(predicted_SASA, SASA_label)
            AA_prediction_loss = loss_fn2(sigmoid(predicted_AA), AA_prediction_label)
            total_loss_batch = ca_denoising_loss * args.ca_denoising_weight + SASA_prediction_loss * args.sasa_pred_weight + AA_prediction_loss * args.AA_prediction_weight

        elif args.whether_SASA == True and args.whether_AA_prediction == False and args.whether_sidec_prediction == False:
            predicted_ca_coord, predicted_SASA = predictor(encoder_embeddings, new_ca_label)
            ca_denoising_loss = loss_fn(predicted_ca_coord, original_ca_label)
            SASA_prediction_loss = loss_fn(predicted_SASA, SASA_label)
            total_loss_batch = ca_denoising_loss * args.ca_denoising_weight + SASA_prediction_loss * args.sasa_pred_weight

        elif args.whether_SASA == False and args.whether_AA_prediction == True and args.whether_sidec_prediction == False:
            predicted_ca_coord, predicted_AA = predictor(encoder_embeddings, new_ca_label)
            ca_denoising_loss = loss_fn(predicted_ca_coord, original_ca_label)
            AA_prediction_loss = loss_fn2(sigmoid(predicted_AA), AA_prediction_label)
            total_loss_batch = ca_denoising_loss * args.ca_denoising_weight + AA_prediction_loss * args.AA_prediction_weight

        elif args.whether_SASA == False and args.whether_AA_prediction == False and args.whether_sidec_prediction == True:
            predicted_ca_coord, predicted_sidec_info = predictor(encoder_embeddings, new_ca_label)
            ca_denoising_loss = loss_fn(predicted_ca_coord, original_ca_label)
            sidec_denoising_loss = loss_fn(predicted_sidec_info, sidec_label)
            total_loss_batch = ca_denoising_loss * args.ca_denoising_weight + sidec_denoising_loss * args.sidec_denoising_weight

        else:
            predicted_ca_coord = predictor(encoder_embeddings, new_ca_label)
            ca_denoising_loss = loss_fn(predicted_ca_coord, original_ca_label)
            total_loss_batch = ca_denoising_loss * args.ca_denoising_weight

        if optimizer:
            total_loss_batch.backward()
            optimizer.step()

        num_nodes = int(batch.mask.sum()) # the perturbed residue number for current batch that is used to calculate the loss
        total_loss += float(total_loss_batch) * num_nodes
        total_count += num_nodes
        # print('average loss of finished batches in the current epoch: %.5f' % float(total_loss / total_count))

        # Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU application and visible invidia-smi.
        # https://blog.csdn.net/weixin_43135178/article/details/117906219
        # clean it based on every batch
        torch.cuda.empty_cache()

    return total_loss / total_count # the average loss over current epoch


if __name__ == '__main__':
    # CA denoising task: based on 'raw' original and new CA coordinates
    # side chain denoising task: based on the original and new coordinates after centroid normalization, thus this task also learns the centroid change due to CA coordinate change
    # if set whether_sidec_noise to False, this task also learns the centroid change due to CA coordinate change and the relationship between side chain atoms and corresponding CA (sidec_CA_relations_label)
    # but if set sidec_denoising_weight = 0, representing that this side chain denoising task will be closed thoroughly

    args = ap.parse_args()
    # if needing to add or change hyper-parameters defined in config.py, just add them like args.epochs = 100 below
    args.data_dir = './data/refer data/PPI/EquiPPI/data/pretraining/'
    args.num_workers = 0 # default: 4
    args.top_k = 20
    args.lr = 0.001
    args.lr_scheduler_patience = 10 # patience for ReduceLROnPlateau scheduler
    args.epochs = 100
    args.noise_type = 'trunc_normal'
    args.noise_prob = 0.15
    args.only_CA = False
    args.whether_sidec_prediction = True # if set whether_sidec_prediction to False, the whole side chain prediction task is banned (not influence sidec features)
    args.whether_sidec_noise = True
    args.sidec_chain_normalization = True
    args.whether_SASA = True
    args.whether_AA_prediction = True
    args.whether_spatial_graph = True # ask pytorch DataLoader generate multi-graph data
    args.graph_cat = 'cat'
    args.early_stop = 30

    args.ca_denoising_weight = 1
    args.sidec_denoising_weight = 1
    args.sasa_pred_weight = 1
    args.AA_prediction_weight = 1
    args.main_num_layers = 5

    for name, value in vars(args).items():
        print(name, value)
    print('node_dim:', node_dim, 'edge_dim:', edge_dim)
    print('*** Above Are All Hyper Parameters ***')

    main(args)
