# MpbPPI
### A multi-task pre-training-based equivariant approach for the prediction of the effect of amino acid mutations on protein-protein interactions

__How to use it:__

__Basic Environment (Windows or Linux):__
* Python 3.9.13
* Pytorch 1.12.1
* CUDA tool kit 11.3.1
* Pytorch Geometric (PyG) for PyTorch 1.12.* and CUDA 11.3


Step1. __Download our pre-processed jsonl file (generated based on the mutant-type (MT) PPI complex source FoldX) from https://drive.google.com/file/d/1Y_3xn5qrnYG79CiQAJKPrw2coL3DSax3/view?usp=sharing, and put it into ./data/ folder as the downstream source data file.__

Step2. __Follow the illustration in \_4_run_MpbPPI_ddg_prediction.py, to read the above jsonl file and run&evaluate MpbPPI for downstream ddG predictions in two different data splitting ways.__

__A running example (including training and evaluation to create the similar results reported in the manuscript):__

python \_4_run_MpbPPI_ddg_prediction.py --data_split_mode 'CV10_random' (mutation-level tenfold cross-validation)

python \_4_run_MpbPPI_ddg_prediction.py --data_split_mode 'complex' (wide-type PPI complex type-based cross-validation)

__The scripts for the full pre-processing pipeline as well as our collected pre-training data and corresponding pre-training script will be released upon acceptance.__



