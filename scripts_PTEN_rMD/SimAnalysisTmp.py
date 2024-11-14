from FoldingAnalysis.analysis import *
import matplotlib.pyplot as plt
import os

###
dirData = "/home/annarita.zanon/ratchet/PTEN/data"
sysName = "PTEN"         # System name

 
i = 1   # Iteration number
n_ratchet = 2    # Number of refolding traj per unfolding traj
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

def RMSD_as_coord(trajectory, selection):
    if selection == "all":
        return trajectory.rmsd[:]
    else:
        return rmsd_traj_selection(trajectory,selection)

def Q_as_coord(trajectory):
    return trajectory.q[:]

def ratchet_to_total_force(run, max_rows=1000):
    """Ratio between ratchet and total force,
    provide a ratchet.out file"""
    data = np.loadtxt(run, max_rows=max_rows)
    ratchet_force = data[:, 4]
    total_force = data[:, 5]
    r_to_t = ratchet_force[1:]/total_force[1:]
    return r_to_t

dirAn = f"{dirData}/tmp/analysis"
if not os.path.exists(dirAn):
    os.system(f"mkdir {dirAn}")

files_to_copy = ["ratchet.out", 
                 "ratchet_md.xtc", 
                 "ratchet_md.tpr", 
                 "box.gro", 
                 "mdout.mdp"]

for file in files_to_copy:
    os.system(f"cp {dirData}/iter_{i}/ratchet_{n_ratchet}/{file} {dirData}/tmp/")
    
os.system(f"gmx trjconv -f {dirData}/tmp/ratchet_md.xtc -s {dirData}/tmp/ratchet_md.tpr -o {dirData}/tmp/ratchet_md_noPBC.xtc -ur compact -center -pbc mol<< EOF \n1\n1\nEOF")

#native = f"{dirData}/../conf/{sysName}.pdb"
native = f"{dirData}/iter_{i}/unfolding/init_conf.gro"


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

traj = f"{dirData}/tmp/ratchet_md_noPBC.xtc"
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
if c_2_type == "RMSD":
    c_2 = RMSD_as_coord(Traj, sel_2)
    y_lab = f"{coord_2} [{units_2}]"
    ax1.set_ylabel(y_lab)
    ax0.set_ylabel(y_lab)
    ax1.set_ylabel(y_lab)
    ax3.set_ylabel(y_lab)

ax0.plot(c_1,c_2,lw=1)

ax1 = axs[0,1]
colors = np.linspace(0, 1, len(c_1))
scatter = ax1.scatter(c_1, c_2, c=colors, cmap=cmap, alpha=0.5, lw=1)

ax2 = axs[1,0]
ax2.plot(range(len(c_1)),c_1,lw=1)

ax3 = axs[1,1]
ax3.plot(range(len(c_2)),c_2,lw=1)

ax1.legend()
ax1.legend(ncol=3)
plt.savefig(f"{dirAn}/refolding.png")

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)
ax1.set_ylabel("Ratchet to total force")
ax1.set_xlabel("Simulation time (ps)")

run = f"{dirData}/tmp/ratchet.out"
r_to_t = ratchet_to_total_force(run, max_rows=len(c_1)-1)
ax1.plot(np.arange(len(r_to_t)),r_to_t,lw=1)

ax1.legend(ncol = 3)
plt.savefig(f"{dirAn}/ratchet_to_total_force.png") 

