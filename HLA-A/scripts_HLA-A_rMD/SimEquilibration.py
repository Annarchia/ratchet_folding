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
forceF = 'charmm36-jul2022'
ssBridge = ''
H2Ofile = 'spc216.gro'

##########
# rMD WT #
##########

sysName = "HLA-A"  # System name
n_unfolded = 1  # Number of unfolding trajectories
unfolding_step = 1
boxL = 1.5  # Simulation box dimension
wDir = "/home/annarita.zanon/ratchet/HLA-A" # Working directory

numThreads = multiprocessing.cpu_count()

scriptsDir = os.getcwd()
os.chdir(wDir)

for i in range(n_unfolded):
    
    iterDir = f"{wDir}/data/iter_{i+1}"
    
    eqDir = f"{iterDir}/equilibration"
    if not os.path.exists(eqDir):
        os.system(f"mkdir {eqDir}")
    
    os.chdir(eqDir)

    if gen_top:
        initConf = f"{iterDir}/unfolding_{unfolding_step}/unfolded_state.pdb"
        os.system(
                f"{gmx} pdb2gmx -f {initConf} -o init_conf.gro -water {water} -ff {forceF} -ignh -ss {ssBridge} -ter << EOF \n1\n0\nEOF"
                )
    else:
        os.system(f"cp {iterDir}/unfolding_{unfolding_step+1}/init_conf.gro {eqDir}")
        os.system(f"cp {iterDir}/unfolding_{unfolding_step+1}/posre.itp {eqDir}")
        os.system(f"head -n -3 {iterDir}/unfolding_{unfolding_step+1}/topol.top > {eqDir}/topol.top")

    os.system(
        f"{gmx} editconf -f init_conf.gro -o box.gro -c yes -bt dodecahedron -d {boxL}"
    )

    os.system(
        f"{gmx} solvate -cp box.gro -o solv.gro -cs {H2Ofile} -p topol.top"
    )

    mdp = f"{mdpDir}/ions.mdp"
    os.system(
        f"{gmx} grompp -f {mdp} -c solv.gro -p topol.top -o solv.tpr"
    )

    os.system(
        f"{gmx} genion -s solv.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral -conc .15 << EOF \n13\nEOF"
    )

    # Energy minimization
    mdp = f"{mdpDir}/em.mdp"
    os.system(
            f"{gmx} grompp -f {mdp} -c ions.gro -p topol.top -o em.tpr"
        )

    os.system(
            f"{gmx} mdrun -v -deffnm em -nt {int(numThreads)}"
        )

    # nvt equilibration
    mdp = f"{mdpDir}/nvt.mdp"
    os.system(
        f"{gmx} grompp -f {mdp} -c em.gro -r em.gro -p topol.top -o nvt.tpr"
        )
    
    os.system(
        f"{gmx} mdrun -deffnm nvt -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
        )
    
    # equilibration MD
    mdp = f"{mdpDir}/equilibration_md.mdp"
    os.system(
        f"{gmx} grompp -f {mdp} -c nvt.gro -r nvt.gro -p topol.top -o equilibration_md.tpr"
        )
    
    os.system(
        f"{gmx} mdrun -deffnm equilibration_md -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
        )
    
    os.system(
        f"{gmx} trjconv -f equilibration_md.xtc -s equilibration_md.tpr -o equilibration_md_noPBC.xtc -ur compact -center -pbc mol<< EOF \n1\n1\nEOF"
        )
    

    traj = mdt.load(f"{eqDir}/equilibration_md_noPBC.xtc"
                    ,top = f"{eqDir}/init_conf.gro")
    
    rmsd = mdt.rmsd(traj
                ,mdt.load(f"{eqDir}/init_conf.gro")
                ,atom_indices = traj.topology.select('backbone'))
    
    time = np.where(rmsd == np.max(rmsd))[0]
    
    traj[time].save_pdb(f"{eqDir}/eq_state.pdb")