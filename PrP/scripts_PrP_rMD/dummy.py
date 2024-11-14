import FoldingAnalysis as fa
import FoldingAnalysis.utilities as utilities
import FoldingAnalysis.analysis as analysis
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import json

with open("../params.json", 'r') as p:
    json_params = json.load(p)

params = json_params['parameters'][0]

sysName = "PrP" #params['sysName']         # System name
n_unfolded = params['n_unfolded']   # Number of unfolding trajectories
n_ratchet = params['n_ratchet']     # Number of refolding traj per unfolding traj
box_l = params['box_l']             # Simulation box dimension

currentDir = f'{os.getcwd()}/..'                                # Working dir

native = f"{currentDir}/{sysName}/data_0/iter_1/unfolding/init_conf.gro"
traj_1 = f"{currentDir}/{sysName}/data_0/iter_1/ratchet_1/md_noPBC.xtc"
traj_2 = f"{currentDir}/{sysName}/data_0/iter_1/ratchet_3/md_noPBC.xtc"
#traj = f"{currentDir}/trajout.xtc"

# Tt = fa.Trajectory(native, traj, reference=fa.Trajectory(native))
# start1 = time.time()
# Qt = Tt.getQ()
# end1 = time.time()

bias_properties = analysis.ratchet_outParser(f"{currentDir}/{sysName}/data_0/iter_1/ratchet_1/ratchet.out")
T1 = analysis.Trajectory(filename=traj_1,ref_filename=native, bias_properties=bias_properties)
bias_properties = analysis.ratchet_outParser(f"{currentDir}/{sysName}/data_0/iter_1/ratchet_3/ratchet.out")
T2 = analysis.Trajectory(filename=traj_2,ref_filename=native, bias_properties=bias_properties)

print(T1.q[:])
print(T1.q_soft[:])
#print(Ta.sarr[0])
Te = analysis.TrajectoryEnsemble(ref_filename=native, trajectories=[T1,T2])
Te.Q_RMSD(file="banana.txt")


#print(T1.ref.sarr[:].shape)
#print(T1.sarr[:].shape)

# T = np.append(T1.sarr[:],[T1.ref.sarr[:]], axis=0)
# print(T)
# print(T.shape)
#print(Ta.u.trajectory.n_frames, end2-start2)
