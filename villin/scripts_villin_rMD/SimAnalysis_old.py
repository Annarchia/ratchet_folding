from FoldingAnalysis.analysis import *
import matplotlib.pyplot as plt
import json
import os
from FoldingAnalysis import energy3Dplot


######
# WT #
######

dirData = "/home/annarita.zanon/ratchet/villin/data"
sysName = "villin"         # System name

n_unfolded = 1   # unfolding trajectory
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

for i in range(n_unfolded, n_unfolded +1):
    
    dirAn = f"{dirData}/iter_{i}/analysis"
    if not os.path.exists(dirAn):
        os.system(f"mkdir {dirAn}")

    native = f"{dirData}/../conf/{sysName}.pdb"
    
    # Unfolding Analysis
    traj = f"{dirData}/iter_{i}/unfolding/unfolding_md_noPBC.xtc"
    Traj = Trajectory(filename=traj, ref_filename=native)

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Thermal Unfolding")
    ax1.set_xlabel("Fraction of native contacts (Q)")
    ax1.set_ylabel("RMSD to native (nm)")
    c_1 = Q_as_coord(Traj)
    c_2 = RMSD_as_coord(Traj, sel_2)

    ax1.plot(c_1,c_2,lw=1)
    plt.savefig(f"{dirAn}/unfolding.png")
    
    fig, axs = plt.subplots(2,2,figsize=(15,12))
    fig.suptitle("Ratchet refolding")
    ax0 = axs[0,0]
    ax1 = axs[0,1]
    ax2 = axs[1,0]
    ax3 = axs[1,1]
    ax2.set_xlabel("Simulation time (ps)")
    ax3.set_xlabel("Simulation time (ps)")
        
    # # Generate a colormap
    cmap = plt.get_cmap('coolwarm')

    Q = []
    RMSD = []
    frames = []
    id_traj = []
    for j in range(n_ratchet):
        traj = f"{dirData}/iter_{i}/ratchet_{j+1}/ratchet_md_noPBC.xtc"
        Traj = Trajectory(filename=traj, ref_filename=native)

        log = f"{dirData}/iter_{i}/ratchet_{j+1}/ratchet_md.log"
        with open(log) as f:
            lines = f.readlines()[-500:]
        for k, line in enumerate(lines[::-1]):
            if line.strip() == "Step           Time":
                t_f = int(lines[-k].split()[1].split(".")[0])
                continue
        
        if c_1_type == "RMSD":
            c_1 = RMSD_as_coord(Traj, sel_1)
            x_lab = f"{coord_1} [{units_1}]"
            ax0.set_xlabel(x_lab)
            ax1.set_xlabel(x_lab)
            ax2.set_ylabel(x_lab)
        if c_1_type == "Q":
            c_1 = Q_as_coord(Traj)
            x_lab = f"{coord_1} (Q)"
            ax0.set_xlabel(x_lab)
            ax1.set_xlabel(x_lab)
            ax2.set_ylabel(x_lab)
            Q.extend(c_1)
        if c_2_type == "RMSD":
            c_2 = RMSD_as_coord(Traj, sel_2)
            y_lab = f"{coord_2} [{units_2}]"
            ax0.set_ylabel(y_lab)
            ax1.set_ylabel(y_lab)
            ax3.set_ylabel(y_lab)
            RMSD.extend(c_2)

        frames.extend(range(len(c_1)))
        id_traj.extend([j+1]*len(c_1))

        ax0.plot(c_1,c_2,lw=1,label=f"run {j+1}")

        ax1 = axs[0,1]
        colors = np.linspace(0, 1, len(c_1))
        scatter = ax1.scatter(c_1, c_2, c=colors, cmap=cmap, alpha=0.5, lw=1, label=f"run {j+1}")

        time = [i / (len(c_1) - 1) * t_f for i in range(len(c_1))]

        ax2 = axs[1,0]
        ax2.plot(time,c_1,lw=1,label=f"run {j+1}")
        
        ax3 = axs[1,1]
        ax3.plot(time,c_2,lw=1,label=f"run {j+1}")

    ax0.legend()
    ax0.legend(ncol=3)
    plt.savefig(f"{dirAn}/refolding.png")

    probability_matrix, allframes_matrix, Qbin, RMSDbin, max_RMSD = energy3Dplot.probability_matrix(np.array(Q), np.array(RMSD), np.array(frames), np.array(id_traj), 25)
    energy3Dplot.probability_plot(probability_matrix, Qbin, RMSDbin, max_RMSD, filename=f"{dirAn}/probability.png")
    energy_matrix, real_values = energy3Dplot.make_matrix_energy(probability_matrix)
    energy3Dplot.energy_plot(energy_matrix, Qbin, RMSDbin, max_RMSD, real_values,filename=f"{dirAn}/energy.png")
    
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel("Ratchet to total force")
    ax1.set_xlabel("Simulation time (ps)")
    
    for j in range(n_ratchet):
        run = f"{dirData}/iter_{i}/ratchet_{j+1}/ratchet.out"
        r_to_t = ratchet_to_total_force(run)
        ax1.plot(np.arange(len(r_to_t)),r_to_t,lw=1,label=f"run {j+1}")
    
    ax1.legend(ncol = 3)
    plt.savefig(f"{dirData}/iter_{i}/analysis/ratchet_to_total_force.png") 

