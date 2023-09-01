# MpbPPI
### A multi-task pre-training-based equivariant approach for the prediction of the effect of amino acid mutations on protein-protein interactions

__Basic Environment Configuation (Windows or Linux, the specific installation time depends on your system configuration):__
* Python 3.9.13
* Pytorch 1.12.1
* CUDA tool kit 11.3.1
* Pytorch Geometric (PyG) for PyTorch 1.12.* and CUDA 11.3

## Steps about how to generate required data for model pre-training and model downstream predictions ##

__In this implementation, the names MpbPPI and EquiPPI of our framework may be used interchangeably__

__Step1. Cleaning data, the aim is to only retain 'ATOM' (but not removing all H atoms here) and 'TER' line in each pdb file for keeping consistency between different datasets.__

For pretraining set: run _1_run_cleaning_pdb.py to clean pdbs (the 'cleaned' suffix will be added to the output folder based on the input folder name).

For downstream sets: same to the above (mainly for widetype pdbs, because the mutation pdbs will be generated based on 'cleaned' widetype pdbs).

__Note:__

1. In this step, we do not consider how to handle amino acid outliers and AALA/BALA cases in PDB files (i.e., there are two sets of coordinates provided for the same residue, but the total number of such cases is not too many).

2. The mutant sample number involving non-natural AAs is 24 covering 5 proteins, and these proteins are only in S4169 downstream set.

__Step2A. Side chain completion, the aim is to complete side chain atoms of cleaned pdbs in pretraining set and downstream widetype(WT) sets__

For pretraining set: run _2_completing_sidechain_pretraining.py (add the 'foldx' suffix to the output folder name).

For downstream sets: run _2_completing_sidechain_downstream.py (add the 'foldx' suffix to the output folder name).

__Note:__

1. For pretraining set, the protein containing any non-natural AAs will be removed (based on the above script).

2. For downstream sets, currently the protein containing any non-natural AAs will not be removed (for the flexibility of next few steps).

3. For both pretraining and downstream sets, in this step, the above AALA/BALA cases will be solved, because after foldx processing, only atoms with Alternate location indicator = A will be retained.

4. Besides, foldx also removes all H atoms, thus after this step, we do not need to remove H atoms again.

5. Current pdbs that cannot be completed by foldx, in S645 and M1101, are pdbs with mutation sites on the residues with numerical serial numbers like 1N8Z.pdb (with residue serial number 100/100A/100B/100C).

6. For downstream sets like S645, there are some pdbs like 1N8Z.pdb with residue serial numbers 100 + 100A + 100B + 100C (but with four different AA types), in the case, these four AAs should be treated as four independent residues to be retained in preprocessing.

7. In the next step, SASA also needs to be calculated independently (based on all heavy atoms and removing H atoms).

8. The generation of mutation structures should be based on the WT pdbs after the FoldX side chain completion, after the generation, to unify the pdb format of widetype and mutation files, i.e., to unify the mutation files that do not include any 'TER' symbols etc., which causes errors on source data json file (as the GEE encoder input) generation (on AA counting and chain number counting) in the next step (i.e., _3_generate_json_input.py), we need to further run _2_run_mutation_unify.py.

__Step2B. Mutant structure generation__

__FoldX:__

1. First please apply for a license for the usage of FoldX 4.0, please refer https://github.com/Liuxg16/GeoPPI and https://foldxsuite.crg.eu/ for the application.

2. The downloaded FoldX binary file should be put into the root dictionary of this repository.

3. We provide a script _5_foldx_mutation_prediction.py (including more detailed description) to generate mutant structure for each WT PDB (after the FoldX side chain completion) in a dataset.

4. In _5_foldx_mutation_prediction.py, we provide an example ("mode == 'calling" + "foldx_mutation_prediction('./data/', 'M1101_foldx_cleaned/', 139)") to demonstrate how FoldX works.

__MODELLER:__

1. The relevant link: https://salilab.org/modeller/10.4/release.html.

2. For MODELLER, the structure for a single mutation or multiple point mutations was modeled based on its wild-type structure using homologous modeling with MODELLER 10.4. We added chain break ('/') characters to the alignment file. Then we used the AutoModel class to build models of the complex.

__AlphaFold2:__

1. The relevant link: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb.

2. For AlphaFold2, we used a locally installed version of the publicly available ColabFold4 to predict mutant structures. The amino acid sequence was modified to reflect the desired mutations, and this updated sequence was used as input for the AlphaFold2. Following the AlphaFold2 pipeline, we performed a MSA search with MMseqs2 on the UniRef30 database5 and generated five models for each sample. These models were subsequently relaxed with the Amber program and ranked. We ultimately selected the top-ranked structure for ddG prediction tasks (all information of above operations can be found in the provided link).

__Step2C. SASA calculation for WT and MT PDB files in each dataset__

1. After WT side chain completion and mutant structure generation, we need to calculate SASA for each WT/MT PDBs as part of the residue node feature (all other features can be generated automatically based on our code).

2. The SASA generation script and corresponding description can be found in ./dasa_demo/get_sasa.sh (based on the GROMACS tool).
    
3. After SASA calculation, we can use original PDB files and corresponding SASA files to generate the input json file for GEE encoder (example of SASA file data structure can be found in ./data/M1101_foldx_cleaned/4_mt_dasa_*).

__Step3. Definition of Pytorch Dataset and Dataloader, the aim is to generate json files containing source data, label, data split for model training and to define corresponding Dataloader for further processing them__

__For pretraining set:__

1. Run _3_generate_json_input.py (task='pretraining', for generating json files).

2. Run _3_datasplit_pretraining.py (for data point training/validation splitting).

3. Call _3_generate_residuefeats_pretraining.py (for defining Dataloader for pretraining) step by step.

__For downstream sets:__

1. Run _3_generate_interface_mutation.py (for generating json files storing mutation site, interface site, and ddg label information based on downstream wildtype proteins).

2. Run _3_generate_json_input.py (task='finetuning', for generating json files), need to ensure file number of WT coordinate, WT SASA, MT coordinate, and MT SASA folders should be the same.

3. Run _3_datasplit_downstream.py (for splitting the downstream set into train/val/test sets based on different modes).

4. Call _3_generate_residuefeats_finetuning.py (for defining Dataloader for finetuning).

__Note:__

1. For pretraining set, there have been no non-natural AAs (as the corresponding proteins are removed in Step2), and in current plan, non-natural AAs in downstream sets will be removed in this step (just remove the non-natural AAs instead of complete proteins).

__Step4. Model training and validation, the aim is to train and validate model based on corresponding data__

For pre-training set: run _4_run_pretraining_MpbPPI\_*.py.

For downstream sets: run _4_run_MpbPPI_ddg_prediction.py.

## Quick start ##

__We have provided our pre-processed source data jsonl files (including the pretraining and downstream S4169 datasets, etc.) in https://drive.google.com/file/d/1nIB3tH0CEYxA1CqFKbA1uCmeecg_8Ni3/view?usp=sharing for a quick start.__ 

Step1. __Download our pre-processed jsonl files in above link, and put it into the ./data/ folder as the source data file.__

Step2. __Follow the illustration in \_4_run_MpbPPI_ddg_prediction.py, to read the above jsonl file and run&evaluate MpbPPI for downstream ddG predictions in two different data splitting ways (for pre-training, the procedure is similar based on the \_4_run_pretraining_MpbPPI_aadenoising_rgraph.py script).__

__A running example (including the training and evaluation to create the similar results reported in the manuscript):__

__After the environment configuration, usually several hours are needed to finish running the demo code.__

python \_4_run_MpbPPI_ddg_prediction.py --data_split_mode 'CV10_random' (mutation-level tenfold cross-validation)

python \_4_run_MpbPPI_ddg_prediction.py --data_split_mode 'complex' (wide-type PPI complex type-based cross-validation)

#### For other files for example the original generated WT/MT PDB files for each downstream dataset (generated by the above pipeline), or any content supporting the conclusion of the manuscript can be provided upon reasonable request. ####











