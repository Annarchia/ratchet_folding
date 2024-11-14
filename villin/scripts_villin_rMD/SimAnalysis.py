from FoldingAnalysis.analysis import *
import matplotlib.pyplot as plt
import json
import os

###
dirData = "/home/annarita.zanon/ratchet/villin/data"
sysName = "villin"         # System name
n_unfolded = 1   # Number of unfolding trajectories
step_unfolding = 1
n_ratchet = 5    # Number of refolding traj per unfolding traj
coord_1 = "RMSD to helix 1"
coord_2 = "RMSD to helix 2"
c_1_type = "RMSD"
c_2_type = "RMSD"
sel_1 =  "resid 42:52"
sel_2 = "resid 63:75"
units_1 = "$\AA$"
units_2 = "$\AA$"

### Change plot style
plt.style.use('seaborn-v0_8-deep')


def RMSD_as_coord(trajectory, selection):
    if selection == "all":
        return trajectory.rmsd[:]
    else:
        return rmsd_traj_selection(trajectory,selection)

def Q_as_coord(trajectory):
    return trajectory.q[:]

for i in range(n_unfolded):
    
    dirAn = f"{dirData}/iter_{i+1}/analysis"
    if not os.path.exists(dirAn):
        os.system(f"mkdir {dirAn}")

    native = f"{dirData}/iter_{i+1}/unfolding_{step_unfolding}/init_conf.gro"
    
    ## Unfolding Analysis
    traj = f"{dirData}/iter_{i+1}/unfolding_{step_unfolding}/unfolding_md_noPBC.xtc"
    Traj = Trajectory(filename=traj, ref_filename=native)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_title("Unfolding")

    if c_1_type == "RMSD":
        c_1 = RMSD_as_coord(Traj, sel_1)
        x_lab = f"{coord_1} [{units_1}]"
        ax1.set_xlabel(x_lab)
    if c_1_type == "Q":
        c_1 = Q_as_coord(Traj)
        x_lab = f"{coord_1} (Q)"
        ax1.set_xlabel(x_lab)
    if c_2_type == "RMSD":
        c_2 = RMSD_as_coord(Traj, sel_2)
        y_lab = f"{coord_2} [{units_2}]"
        ax1.set_ylabel(y_lab)

    ax1.plot(c_1,c_2,lw=1)
    plt.savefig(f"{dirAn}/unfolding.png")

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)

    ax1.set_title("Ratchet refolding")
    ax1.set_xlabel(x_lab)
    ax1.set_ylabel(y_lab)
    
        
    # Generate a colormap
    cmap = plt.get_cmap('coolwarm')

    for j in range(n_ratchet):
        traj = f"{dirData}/iter_{i+1}/ratchet_{j+1}/ratchet_md_noPBC.xtc"
        Traj = Trajectory(filename=traj, ref_filename=native)

        if c_1_type == "RMSD":
            c_1 = RMSD_as_coord(Traj, sel_1)
        if c_1_type == "Q":
            c_1 = Q_as_coord(Traj)
        if c_2_type == "RMSD":
            c_2 = RMSD_as_coord(Traj, sel_2)
    
        ax1.plot(c_1,c_2,lw=1,label=f"run {j+1}")

    ax1.legend()
    ax1.legend(ncol=3)
    plt.savefig(f"{dirAn}/refolding.png")
    
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel("Ratchet to total force")
    ax1.set_xlabel("Simulation time (ps)")
    
    for j in range(n_ratchet):
        run = f"{dirData}/iter_{i+1}/ratchet_{j+1}/ratchet.out"
        r_to_t = ratchet_to_total_force(run)
        ax1.plot(np.arange(len(r_to_t)),r_to_t,lw=1,label=f"run {j+1}")
    
    ax1.legend(ncol = 3)
    plt.savefig(f"{dirAn}/ratchet_to_total_force.png") 

