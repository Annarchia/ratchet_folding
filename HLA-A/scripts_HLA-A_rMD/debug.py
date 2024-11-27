import os
from FoldingAnalysis.analysis import *
import FoldingAnalysis.utilities as utilities
import numpy as np
import math
import json
import sys

### Load user-defined parameters

with open("../params.json", 'r') as p:
    json_params = json.load(p)

params = json_params['parameters'][0]

sysName = params['sysName']         # System name
n_unfolded = params['n_unfolded']   # Number of unfolding trajectories
n_ratchet = params['n_ratchet']     # Number of refolding traj per unfolding traj


w_dir = f'{os.getcwd()}/../../prova/HLA-A/data/iter_1'
ref_pdb_name = f'{w_dir}/../../conf/{sysName}.pdb'
o_dir = w_dir + '/AverageCmaps'

try: 
    os.makedirs(o_dir, exist_ok=True)
except OSError as e:
    print('ERROR: Cannot create result directory!')
    raise

reference = Structure(filename_or_universe=ref_pdb_name, name='reference')
#reference.configureCmap(selection='all')

trajectories = []
print('> Opening trajectories...')
for f in os.listdir(w_dir):
    if not f.startswith('ratchet'):
        continue
    i_traj = f'{w_dir}/{f}/md_trimmed.xtc'
    if not os.path.isfile(i_traj):
        continue
    trajectories.append(Trajectory(filename=i_traj, ref_filename=ref_pdb_name))

ensemble = TrajectoryEnsemble(ref_filename=ref_pdb_name, trajectories=trajectories, dt=1)
    #ensemble.configureFolding(method='rmse',ignore_last_time=100,tolerance=0.3, threshold=2.5)
    #ensemble.configureCmap(selection='all', use_ref_as_last=False) # step=5)
print('> Done')


print(f'> Folding Trajectories: {ensemble.getFoldedTrajectoriesCount()}')

    #average_cmaps = np.memmap(f'{o_dir}/average.mmp', dtype=np.float32, mode='w+', shape=(n_frames, ensemble.getFoldedTrajectories()[0].getCmapDim()))
ensemble.getAverageCmapTime(folded_only=False, stride=10, path=o_dir)
average_cmaps = np.memmap(f'{o_dir}/average_cmaps.mmap', dtype='float64', mode='r')
# Now you can use average_cmaps like a regular numpy array
print(average_cmaps)