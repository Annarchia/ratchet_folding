import os
from FoldingAnalysis.analysis import *
import multiprocessing
import subprocess
import time
import signal



gmx = "gmx"
mdpDir = "/home/annarita.zanon/ratchet/mdp_files"
gen_top = False
water = 'tip3p'
forceF = 'charmm36-jul2022'
ssBridge = ''
H2Ofile = 'spc216.gro'

##########
# rMD WT #
##########

# sysName = "SYS"  # System name
# i = 1
# n_unfolded = 1 # Number of unfolding trajectories
# n_ratchet = 1 # Number of refolding traj per unfolding
# boxL = 1  # Simulation box dimension
# targetConf = "/home/annarita.zanon/ratchet/SYS/conf/SYS.pdb"
# wDir = "/home/annarita.zanon/ratchet/SYS" # Working directory
# topDir = f"/home/annarita.zanon/ratchet/SYS/data/iter_{i+1}/ratchet_init"

#############
# rMD Y178D #
#############

sysName = "SYS_Y178D"  # System name
i = 1
n_unfolded = 1 # Number of unfolding trajectories
n_ratchet = 3 # Number of refolding traj per unfolding
boxL = 1  # Simulation box dimension
targetConf = "/home/annarita.zanon/ratchet/SYS_Y178D/conf/SYS_Y178D.pdb"
wDir = "/home/annarita.zanon/ratchet/SYS_Y178D" # Working directory
topDir = f"/home/annarita.zanon/ratchet/SYS_Y178D/data/iter_{i+1}/ratchet_init"

# Function to monitor `ratchet.out` for the stopping condition
def monitor_ratchet_output(ratchet_file_path, process, sleep_time=60*20):
    while True:
        # Check if the process is still running
        if process.poll() is not None:
            print("Simulation process ended.")
            break  # Exit if the process has ended
        
        if os.path.exists(ratchet_file_path):
            with open(ratchet_file_path, 'r') as file:
                try:
                    line = file.readlines()[-2]
                    if line.startswith('#'):
                        continue
                except:
                    continue
                columns = line.split()
                try:
                    cum_rat_f = float(columns[4])
                    cum_tot_f = float(columns[5])
                    if cum_tot_f > 0 and cum_rat_f / cum_tot_f > 20:
                        print(f"Stopping simulation: CumRatF/CumTotF = {round(cum_rat_f / cum_tot_f, 2)}")
                        process.terminate()  # Terminate the simulation process
                        return
                    else:
                        print(f"Continuing simulation: CumRatF/CumTotF = {round(cum_rat_f / cum_tot_f, 2)}")
                except (IndexError, ValueError):
                    continue
        time.sleep(sleep_time)  # Check every 5 seconds


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
        initConf = f"{iterDir}/ratchet_init/unfolded_state.pdb"
        os.system(
                f"{gmx} pdb2gmx -f {initConf} -o init_conf.gro -water {water} -ff {forceF} -ignh -ss {ssBridge} -ter << EOF \n1\n0\nEOF"
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
    
    # Start the simulation process
    mdrun_command = f"{gmx} mdrun -deffnm ratchet_md -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
    mdrun_process = subprocess.Popen(mdrun_command, shell=True)

    # Monitor the ratchet.out file and stop the simulation if needed
    ratchet_output_path = f"{ratchetDir}/ratchet.out"
    monitor_ratchet_output(ratchet_output_path, mdrun_process)
    
    os.system(
        f"{gmx} trjconv -f ratchet_md.xtc -s ratchet_md.tpr -o ratchet_md_noPBC.xtc -ur compact -center -pbc mol<< EOF \n1\n1\nEOF"
        )