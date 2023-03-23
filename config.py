import argparse


ap = argparse.ArgumentParser(description='Common hyper-parameters for MpbPPI pretraining and finetuning')

ap.add_argument('--data-dir', default='./data/',
                help='the path to store original json data and data split file')
ap.add_argument('--num-workers', type=int, default=4, help='number of threads for loading data, default = 4')
ap.add_argument('--max-nodes', type=int, default=3000, help='max number of nodes per batch, default=3000')
ap.add_argument('--top-k', type=int, default=20, help='K value for the KNN graph of each protein')
ap.add_argument('--lr', type=float, default=0.001, help='initial learning rate of the gvp optimizer')
ap.add_argument('--lr-scheduler-patience', type=int, default=10, help='patience for learning rate, default=10')
ap.add_argument('--epochs', type=int, default=100, help='training epochs, default=100')
ap.add_argument('--noise-type', type=str, default='trunc_normal',
                help='noise type to be added into original coordinates, choice={trunc_normal, normal, uniform}')
ap.add_argument('--noise-prob', type=float, default=0.15,
                help='the probability for adding noise to each residue for a protein, default=0.15')
ap.add_argument('--only-CA', type=bool, default=False, help='whether only to add noise to CA rather than all backbone atoms')
ap.add_argument('--whether-sidec-noise', type=bool, default=True, help='whether to add the same type noise as CA coordinates to side chain atoms')
ap.add_argument('--whether-SASA', type=bool, default=True,
                help='whether to use SASA prediction as an auxiliary task in pretraining')
ap.add_argument('--ca-denoising-weight', type=float, default=1, help='loss weight ratio for CA denoising task')
ap.add_argument('--sidec-denoising-weight', type=float, default=1,
                help='loss weight ratio for side chain coordinate information denoising task')
ap.add_argument('--sasa-pred-weight', type=float, default=1,
                help='loss weight ratio for SASA prediction task (if applicable, which is set by whether-SASA)')
ap.add_argument('--valid-steps', type=int, default=1, help='how many epochs does model run one time validation process')
ap.add_argument('--early-stop', type=int, default=30, help='the number of epochs for early stopping')
ap.add_argument('--main-num-layers', type=int, default=5, help='number of layers of the equivariant gvp, default = 3-6')
ap.add_argument('--main-dropout', type=float, default=0.1, help='dropout value for the equivarianr gvp, default=0.1')

# for multi-task pretraining predictors
ap.add_argument('--aux-layer-list', default=[512, 128, 3],
                help='layer neuron units list (except for input unit number) for the multi-task pretraining predictor')
ap.add_argument('--aux-in-dropout', type=float, default=0.2,
                help='The input layer dropout rate of the multi-task pretraining predictor')
ap.add_argument('--aux-med-dropout', type=float, default=0.2,
                help='The intermediate layer dropout rate of the multi-task pretraining predictor')

# for downstream finetuning decoder
ap.add_argument('--down-layer-list', default=[512, 128, 1],
                help='layer neuron units list (except for input unit number) for the downstream finetuning decoder')
ap.add_argument('--down-in-dropout', type=float, default=0.2,
                help='The input layer dropout rate of the downstream finetuning decoder')
ap.add_argument('--down-med-dropout', type=float, default=0.2,
                help='The intermediate layer dropout rate of the downstream finetuning decoder')