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


w_dir = f'{os.getcwd()}/../../prova/HLA-A/data/iter_1/'
ref_pdb_name = f'{w_dir}/../../conf/{sysName}.pdb'
o_dir = w_dir + '/AverageCmaps'

try: 
    os.makedirs(o_dir, exist_ok=True)
except OSError as e:
    print('ERROR: Cannot create result directory!')
    raise

reference = Structure(filename_or_universe=ref_pdb_name, name='reference')
#reference.configureCmap(selection='all')
try:
    c_q = np.loadtxt(f'{o_dir}/c_q.txt')
    average_cmaps = np.memmap(f'{o_dir}/average.mmp', dtype=np.float32, mode='r', shape=(c_q.shape[0], reference.getCmapDim()))
    print('> Loaded cached files, skipping cmap claculation')
except (FileNotFoundError, OSError):
    trajectories = []
    print('> Opening trajectories...')
    for f in os.listdir(w_dir):
        if not f.startswith('ratchet'):
            continue
        i_traj = f'{w_dir}/{f}/md_noPBC.xtc'
        if not os.path.isfile(i_traj):
            continue
        trajectories.append(Trajectory(filename=i_traj, ref_filename=ref_pdb_name))

    ensemble = TrajectoryEnsemble(ref_filename=ref_pdb_name, trajectories=trajectories, dt=1)
    #ensemble.configureFolding(method='rmse',ignore_last_time=100,tolerance=0.3, threshold=2.5)
    #ensemble.configureCmap(selection='all', use_ref_as_last=False) # step=5)
    print('> Done')

    if not ensemble.getFoldedTrajectoriesCount() > 0:
        print('NO FOLDING TRAJECTORY!')
        sys.exit(1)

    print(f'> Folding Trajectories: {ensemble.getFoldedTrajectoriesCount()}')
    mean_folding_time = ensemble.meanFoldingTime()
    print(f'> Average folding time: {mean_folding_time}')
    print('> Computing average over time...')
    #ensemble.configureCmap(selection='protein and (resid 868:909 or resid 697:713)', end=int(mean_folding_time+1))
    n_frames = ensemble.getFoldedTrajectories()[0].getFrameCount()
    #average_cmaps = np.memmap(f'{o_dir}/average.mmp', dtype=np.float32, mode='w+', shape=(n_frames, ensemble.getFoldedTrajectories()[0].getCmapDim()))
    average_cmaps = ensemble.getAverageCmapTime(stride=100)
    # utilities.saveCmapNew(o_dir +  '/avg_comp.cmp',
    #                           average_cmaps,
    #                           reference.getCmapSignature(),
    #                           np.ones(average_cmaps.shape[0]),
    #                           np.arange(average_cmaps.shape[0]))

    c_q = utilities.cmap2hard_q(average_cmaps, reference.sarr[:])
    np.savetxt(o_dir + '/c_q.txt',c_q)

# cut_index_last = np.where(c_q > 0.8)[0][0] + 1
# cut_index_first = np.where(c_q < 0.15)[0][-1]
# print('> Cut index first: ' + str(cut_index_first))
# print('> Cut index last: ' + str(cut_index_last))
cut_index_last = average_cmaps.shape[0]
cut_index_first = 0
downsampled_cmaps = None
indexes = None
print('> Done')
f_dist = 0.084
while f_dist > 0:
    dist = input('> Enter distance for downsample, -1 to save:')
    try:
        f_dist = float(dist)
        if f_dist > 0:
            downsampled_cmaps, indexes = utilities.downsampleOverDistance(average_cmaps[cut_index_first:cut_index_last], reference.getCmap(cache=True)[0], distance=f_dist, margin_factor=0.5, rev=True, return_index=True)
            print('> Downsampled cmaps: '+str(len(downsampled_cmaps)))
    except:
        pass 

print('> Indexes: '+str(indexes))
norm_indexes = np.interp(indexes,(indexes.min(),indexes.max()),(0,1))
print('> Norm Indexes: '+str(norm_indexes))

if downsampled_cmaps is not None:
    print('> Saving average cmaps...')
    lambdas = np.empty(downsampled_cmaps.shape[0])
    lambdas[1:] = np.linalg.norm(downsampled_cmaps[:-1] - downsampled_cmaps[1:], axis=1)**2
    lambdas[0] = lambdas[1]
    utilities.saveCmapNew(o_dir +  '/avg.cmp',
                          downsampled_cmaps,
                          reference.getCmapSignature(),
                          1/lambdas,
                          norm_indexes)
    print('> Done')

# if indexes is not None:
#     np.save(o_dir +  '/indexes',indexes)