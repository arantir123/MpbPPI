# MpbPPI
### A multi-task pre-training-based equivariant approach for the prediction of the effect of amino acid mutations on protein-protein interactions

__How to use it:__

Basic Environment:
* __Python 3.9.13__
* __Pytorch 1.12.1__
* __CUDA tool kit 11.3.1__
* __Pytorch Geometric (PyG) for PyTorch 1.12.* and CUDA 11.3__


Step1. __Download our pre-processed jsonl file from https://drive.google.com/file/d/1Y_3xn5qrnYG79CiQAJKPrw2coL3DSax3/view?usp=sharing, and put it into ./data/ folder as the source data file.__

Step2. __Follow the illustration in \_4_run_MpbPPI_ddg_prediction.py, to read the above jsonl file and run&evaluate MpbPPI for downstream ddG predictions in two different data splitting ways.__

__A running example (including training and evaluation):__

python \_4_run_MpbPPI_ddg_prediction.py --data_split_mode 'CV10_random' (mutation-level tenfold cross-validation)

python \_4_run_MpbPPI_ddg_prediction.py --data_split_mode 'complex' (widetype PPI complex type-based cross-validation)

__The scripts for the full preprocessing pipeline as well as our collected pre-training data and corresponding training script will be released upon acceptance.__



