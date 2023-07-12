from Bio.PDB import PDBParser, PDBIO
import sys

io = PDBIO()
_fpdb=sys.argv[1]
_pdbname=_fpdb.split(".pdb")[0]
pdb = PDBParser().get_structure(_pdbname,_fpdb)
print(_fpdb)

_count_chain = 0
for chain in pdb.get_chains():
    io.set_structure(chain)
    io.save(pdb.get_id() +"." + chain.get_id() +".pdb")
    print(pdb.get_id() +"." + chain.get_id() +".pdb")
    _count_chain = _count_chain + 1 

if _count_chain == 1:
    fa=open("single_chain.txt",'a')
    fa.write(_fpdb)
    fa.write("\n")
    fa.close
