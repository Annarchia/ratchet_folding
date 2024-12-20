import numpy as np
import os
import json
import mdtraj as mdt

from FoldingAnalysis.analysis import *

import multiprocessing


gmx = "gmx"
mdpDir = "/home/annarita.zanon/ratchet/mdp_files_PTEN"
gen_top = False
water = 'tip3p'
forceF = ''
ssBridge = ''
H2Ofile = 'spc216.gro'

##########
# rMD WT #
##########

sysName = "HLA-A"  # System name
n_unfolded = 1 # Number of unfolding trajectories
unfolding_step = 1 # Number of unfolding steps
boxL = 4  # Simulation box dimension
targetConf = "/home/annarita.zanon/ratchet/HLA-A/conf/HLA-A.pdb"
wDir = "/home/annarita.zanon/ratchet/HLA-A" # Working directory

numThreads = multiprocessing.cpu_count()

# Create the folder to store the results if it does not exist
if not os.path.exists(f"{wDir}/data"):
    os.system(f"mkdir {wDir}/data")

scriptsDir = os.getcwd()
os.chdir(wDir)

for i in range(n_unfolded):
    
    iterDir = f"{wDir}/data/iter_{i+1}"
    if not os.path.exists(iterDir):
        os.system(f"mkdir {iterDir}")
    
    unfoldDir = f"{iterDir}/unfolding_{unfolding_step}"
    if not os.path.exists(unfoldDir):
        os.system(f"mkdir {unfoldDir}")
    
    Structure(targetConf).save_sarr(f'{unfoldDir}/target.cmp')

    os.chdir(unfoldDir)

    # if gen_top:
    #     initConf = f"{wDir}/conf/{sysName}.pdb"
    #     os.system(
    #             f"{gmx} pdb2gmx -f {initConf} -o init_conf.gro -water {water} -ff {forceF} -ignh -ss {ssBridge} -ter << EOF \n1\n0\nEOF"
    #             )
    # else:
    #     if unfolding_step == 1:
    #         os.system(f"cp {wDir}/data/init_conf.gro {unfoldDir}")
    #         os.system(f"cp {wDir}/data/posre.itp {unfoldDir}")
    #         os.system(f"cp {wDir}/data/topol.top {unfoldDir}")

    # os.system(
    #     f"{gmx} editconf -f init_conf.gro -o box.gro -c yes -bt dodecahedron -d {boxL}"
    # )

    # os.system(
    #     f"{gmx} solvate -cp box.gro -o solv.gro -cs {H2Ofile} -p topol.top"
    # )

    # mdp = f"{mdpDir}/ions.mdp"
    # os.system(
    #     f"{gmx} grompp -f {mdp} -c solv.gro -p topol.top -o solv.tpr"
    # )

    # os.system(
    #     f"{gmx} genion -s solv.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral -conc .15 << EOF \n13\nEOF"
    # )

    # # Energy minimization
    # mdp = f"{mdpDir}/em.mdp"
    # os.system(
    #         f"{gmx} grompp -f {mdp} -c ions.gro -p topol.top -o em.tpr"
    #     )

    # os.system(
    #         f"{gmx} mdrun -v -deffnm em -nt {int(numThreads)}"
    #     )

    # # nvt equilibration
    # mdp = f"{mdpDir}/nvt.mdp"
    # os.system(
    #     f"{gmx} grompp -f {mdp} -c em.gro -r em.gro -p topol.top -o nvt.tpr"
    #     )
    
    # os.system(
    #     f"{gmx} mdrun -deffnm nvt -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
    #     )
    
    # unfolding MD
    mdp = f"{mdpDir}/ratchet_unfolding_md.mdp"
    os.system(
        f"{gmx} grompp -f {mdp} -c nvt.gro -r nvt.gro -p topol.top -o unfolding_md.tpr"
        )
    
    os.system(
        f"{gmx} mdrun -deffnm unfolding_md -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
        )
    
    os.system(
        f"{gmx} trjconv -f unfolding_md.xtc -s unfolding_md.tpr -o unfolding_md_noPBC.xtc -ur compact -center -pbc mol<< EOF \n1\n1\nEOF"
        )
    

    traj = mdt.load(f"{unfoldDir}/unfolding_md_noPBC.xtc"
                    ,top = f"{unfoldDir}/init_conf.gro")
    
    rmsd = mdt.rmsd(traj
                ,mdt.load(f"{unfoldDir}/init_conf.gro")
                ,atom_indices = traj.topology.select('backbone'))
    
    time = np.where(rmsd == np.max(rmsd))[0]
    
    traj[time].save_pdb(f"{unfoldDir}/unfolded_state.pdb")