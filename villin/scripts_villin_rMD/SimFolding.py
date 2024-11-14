import numpy as np
import os
import json
from FoldingAnalysis.analysis import *
import multiprocessing
import FoldingAnalysis as fa
import FoldingAnalysis.utilities as utilities


gmx = "gmx"
mdpDir = "/home/annarita.zanon/ratchet/mdp_files_villin"
gen_top = True
water = 'tip3p'
forceF = 'amber99sb-ildn'
ssBridge = ''
H2Ofile = 'spc216.gro'

##########
# rMD WT #
##########

sysName = "villin"  # System name
i = 0
n_unfolded = 1 # Number of unfolding trajectories
unfold_step = 1
n_ratchet = 5 # Number of refolding traj per unfolding
boxL = 1  # Simulation box dimension
targetConf = "/home/annarita.zanon/ratchet/villin/conf/villin.pdb"
wDir = "/home/annarita.zanon/ratchet/villin" # Working directory
topDir = f"/home/annarita.zanon/ratchet/villin/data/iter_{n_unfolded}/unfolding_{unfold_step}"

numThreads = multiprocessing.cpu_count()

scriptsDir = os.getcwd()
os.chdir(wDir)

for j in range(n_ratchet):
    
    iterDir = f"{wDir}/data/iter_{i+1}"
    
    ratchetDir = f"{iterDir}/ratchet_{j+1}"
    if not os.path.exists(ratchetDir):
        os.system(f"mkdir {ratchetDir}")

    Structure(targetConf).save_sarr(f'{ratchetDir}/target.cmp')

    os.chdir(ratchetDir)

    if gen_top:
        initConf = f"{topDir}/unfolded_state.pdb"
        os.system(
                f"{gmx} pdb2gmx -f {initConf} -o init_conf.gro -water {water} -ff {forceF} -ignh -ss {ssBridge}"
                #f"{gmx} pdb2gmx -f {initConf} -o init_conf.gro -water {water} -ff {forceF} -ignh -ss {ssBridge} -ter << EOF \n1\n0\nEOF"
                )
    else:
        os.system(f"cp {topDir}/init_conf.gro {ratchetDir}")
        os.system(f"cp {topDir}/posre.itp {ratchetDir}")
        os.system(f"cp {topDir}/topol.top {ratchetDir}")

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
    
    # npt equilibration
    mdp = f"{mdpDir}/npt.mdp"
    os.system(
        f"{gmx} grompp -f {mdp} -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr"
        )
    
    os.system(
        f"{gmx} mdrun -deffnm npt -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
        )
    
    # unfolding MD
    mdp = f"{mdpDir}/ratchet_md.mdp"
    os.system(
        f"{gmx} grompp -f {mdp} -c npt.gro -r npt.gro -p topol.top -o ratchet_md.tpr"
        )
    
    os.system(
        f"{gmx} mdrun -deffnm ratchet_md -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
        )
    
    os.system(
        f"{gmx} trjconv -f ratchet_md.xtc -s ratchet_md.tpr -o ratchet_md_noPBC.xtc -ur compact -center -pbc mol<< EOF \n1\n1\nEOF"
        )