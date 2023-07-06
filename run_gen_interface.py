import glob, os, random
import sys
from pymol import cmd
import InterfaceResidues
import shutil


# based on pymol, current pdb name is based on the name in pre-processed downstream set
def gen_inferface(foldx_folder, pdbfile, if_info, workdir):
    # 1PPF.pdb E_I temp
    # print('pdbfile, if_info, workdir:', pdbfile, if_info, workdir) # pdbfile name，Format of [partnerA_partnerB], output path

    pdbobject = pdbfile
    namepdb = os.path.basename(pdbobject)
    # print(namepdb) # os.path.basename() method returns the tail part after splitting the specified path into (head, tail) pair.

    name = namepdb.split('.')[0] # just pdb name (not including .pdb)

    interface_info = if_info # partner information, e.g., AB_CD, indicating AB and CD chains belong to two separate proteins to interact
    chainsAB = interface_info.split('_')
    chainsAB = chainsAB[0] + chainsAB[1] # interactive chain a and chain b

    workdir = workdir
    # use pymol.cmd to load current pdb complex
    # ./test_demo/12as.pdb
    cmd.load(foldx_folder + pdbobject)

    interfaces = []
    # print('chainsAB:', chainsAB)
    # print('name:', name)

    for i in range(len(chainsAB)):
        for j in range(i+1, len(chainsAB)):
            cha, chb = chainsAB[i], chainsAB[j] # 'AB_CD'->'ABCD': A B, A C, A D, B C, B D, C D, iterate all chain combinations in current pdb
            if cha == chb: continue

            # Pymol interfaceResidue finds interface residues between two proteins or chains, using the following concept.
            # First, we take the area of the complex. Then, we split the complex into two pieces, one for each chain. Next, we calculate the chain-only surface area.
            # Lastly, we take the difference between the comeplex-based areas and the chain-only-based areas. If that value is greater than your supplied cutoff, then we call it an interface residue.
            # print('name, cha, chb:', name, cha, chb)
            cmd.do('interfaceResidue {}, chain {}, chain {}'.format(name, cha, chb)) # temp/temp.txt is generated from interfaceResidue
            # Interface residues of a protein are the residues that contact with residues from the interacting proteins. Protein core residues are the non-interface residues whose relative solvent accessibility (rASA) is less than 25%. Non-interface surface residues are the non-interface residues whose rASA is at least 25%

            mapp = {'chA':cha, 'chB':chb}
            # modification:
            # ffile = open('temp/temp.txt','r')
            with open('temp/temp.txt','r') as ffile:
                for line in ffile.readlines():
                    linee = line.strip().split('_') # 20_chA
                    resid = linee[0] # residue id
                    chainn = mapp[linee[1]] # chain id
                    inter = '{}_{}_{}_{}'.format(cha, chb, chainn, resid)
                    if inter not in interfaces:
                        interfaces.append(inter) # chain a + chain b + interface chain id + interface residue id

            # print(interfaces) # obtain the interface information of this complex
            # ['A_B_A_1', 'A_B_A_32', 'A_B_A_34', 'A_B_A_36']
            # os.system('rm temp/temp.txt')
            os.remove('./temp/temp.txt')

    # ffile = open('{}/interface.txt'.format(workdir),'w')
    # for x in interfaces:
    #     ffile.write(x + '\n')

    # print('pdbobject:', pdbobject)
    # cmd.save('{}/'.format(workdir) + pdbobject) # re-write to original pdb file, it seems that after the processing by pymol, the format of pdb file will be changed slightly
    cmd.delete('all') # delete cache
    return interfaces


if __name__ == '__main__':
    # it seems that even though the residue serial number is added 1000, pymol still can output correct interface positions (compared with 12as_1.pdb in pretraining dataset)
    # gen_inferface('./test_demo/', '12as_1.pdb', 'A_B', 'temp')
    gen_inferface('./test_demo/', '1DVF.pdb', 'AB_CD', 'temp')