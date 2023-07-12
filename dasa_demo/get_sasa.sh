
### GROMACS to be installed in your machine. (https://www.gromacs.org/tutorial_webinar.html). GROMACS is a free and open-source software suite for high-performance molecular dynamics and output analysis.

source ~/Work/softwares/Gromacs/gromacs-2022.1/bin/GMXRC   ## Source GMXRC to get access to GROMACS


pdb=$1   ### Input pdbfile 

python get_chains.py $pdb > chn.xvg   ### It separates the complex into different parts, one for each chain.

for line in $(cat chn.xvg)
do
echo $line

## SASA caculation. Detail please check https://manual.gromacs.org/current/onlinehelp/gmx-sasa.html.
## It calculates the surface area of each amino acid of the complex and each chain independently. The average and standard deviation of the area over the pdb can be calculated per residue (options -or).
gmx_mpi sasa -f $line -s $line -or $line.xvg << EOF      
2
EOF
sed -i '1,25d' $line.xvg
rm -rf \#*
done 

python get_dasa.py chn.xvg $pdb > $pdb.sasa   ## It calculates the difference in the surface area of each amino acid in the complex and in each chain.
