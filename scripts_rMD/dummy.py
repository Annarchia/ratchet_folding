from FoldingAnalysis.analysis import *
import matplotlib.pyplot as plt
import json
import os
from scipy.ndimage import gaussian_filter
from FoldingAnalysis import energy3Dplot


#########
# Y178D #
#########

dirData = "/home/annarita.zanon/ratchet/SYS_Y178D/data"
sysName = "SYS_Y178D"         # System name

n_unfolded = 1   # Number of unfolding trajectories
step_unfolding = 1
n_ratchet = 1   # Number of refolding traj per unfolding traj
coord_1 = "Fraction of native contacts"
coord_2 = "RMSD to native"
c_1_type = "Q"
c_2_type = "RMSD"
sel_1 =  "protein"
sel_2 = "protein"
units_1 = "Q"
units_2 = "nm"

### Change plot style
plt.style.use('seaborn-v0_8-deep')

currentDir = f'{os.getcwd()}/..'

def RMSD_as_coord(trajectory, selection):
    if selection == "all":
        return trajectory.rmsd[:]
    else:
        return rmsd_traj_selection(trajectory,selection)

def Q_as_coord(trajectory):
    return trajectory.q[:]

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

for i in range(n_unfolded):
    
    dirAn = f"{dirData}/iter_{i+1}/analysis"
    if not os.path.exists(dirAn):
        os.system(f"mkdir {dirAn}")

    native = f"{dirData}/../conf/{sysName}.pdb"

    fig, axs = plt.subplots(2,2,figsize=(15,12))
    fig.suptitle("Ratchet refolding")
    ax0 = axs[0,0]
    ax1 = axs[0,1]
    ax2 = axs[1,0]
    ax3 = axs[1,1]
    ax2.set_xlabel("Simulation time (ps)")
    ax3.set_xlabel("Simulation time (ps)")
        
    # Generate a colormap
    cmap = plt.get_cmap('coolwarm')

    Q = []
    RMSD = []
    frames = []
    id_traj = []
    for j in range(n_ratchet):
        traj = f"{dirData}/iter_{i+1}/ratchet_{j+1}/ratchet_md_noPBC.xtc"
        Traj = Trajectory(filename=traj, ref_filename=native)


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

        ax2 = axs[1,0]
        ax2.plot(range(len(c_1)),c_1,lw=1,label=f"run {j+1}")
        
        ax3 = axs[1,1]
        ax3.plot(range(len(c_2)),c_2,lw=1,label=f"run {j+1}")


    ax0.legend()
    ax0.legend(ncol=3)
    plt.savefig(f"{dirAn}/refolding.png")

    probability_matrix, allframes_matrix, Qbin, RMSDbin, max_RMSD = energy3Dplot.probability_matrix(np.array(Q), np.array(RMSD), np.array(frames), np.array(id_traj), 25)
    energy3Dplot.probability_plot(probability_matrix, Qbin, RMSDbin, max_RMSD, filename="banana_probability.png")
    energy_matrix, real_values= energy3Dplot.make_matrix_energy(probability_matrix)
    energy3Dplot.energy_plot(energy_matrix, Qbin, RMSDbin, max_RMSD, real_values,filename="babana_energy.png")

    
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel("Ratchet to total force")
    ax1.set_xlabel("Simulation time (ps)")
    
    for j in range(n_ratchet):
        run = f"{currentDir}/{sysName}/data/iter_{i+1}/ratchet_{j+1}/ratchet.out"
        r_to_t = ratchet_to_total_force(run)
        ax1.plot(np.arange(len(r_to_t)),r_to_t,lw=1,label=f"run {j+1}")
    
    r_to_t = r_to_t[:100]
    ax1.plot(np.arange(len(r_to_t)),r_to_t,lw=1,label=f"run {j+1}")
    ax1.legend(ncol = 3)
    plt.savefig(f"dummy.png") 

