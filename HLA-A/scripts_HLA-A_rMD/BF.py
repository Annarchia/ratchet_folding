import os
import json

import FoldingAnalysis.analysis as analysis

## Load user-defined parameters

with open('/home/annarita.zanon/ratchet/params.json', 'r') as p:
    json_params = json.load(p)

params = json_params['parameters'][0]

sysName = params['sysName']         # System name
n_unfolded = params['n_unfolded']   # Number of unfolding trajectories
n_ratchet = params['n_ratchet']     # Number of refoldings per unfolding

# Working dir
refPdb = f'/home/annarita.zanon/ratchet/{sysName}/conf/{sysName}.pdb'   # Reference struct

for iteration in range(n_unfolded):
    dirData = f'/home/annarita.zanon/ratchet/{sysName}/data/iter_{iteration + 1}'
    dirAn = f'{dirData}/analysis'

    os.system(f'''
              if [ ! -d {dirAn}/RMSD ]; 
              then mkdir {dirAn}/RMSD; 
              else rm -rf .{dirAn}/RMSD/*; fi;
              ''') 

    trajectories = []

    print('> Opening trajectories...')
    for f in os.listdir(dirData):
        if not f.startswith('ratchet'):
            continue

        i_traj = f'{dirData}/{f}/md_noPBC.xtc'
        i_rathcet = f'{dirData}/{f}/ratchet.out'

        if not os.path.isfile(i_traj) or not os.path.isfile(i_rathcet):
            continue
        bias_sproperties = analysis.ratchet_outParser(i_rathcet)
        trajectories.append(analysis.Trajectory(filename=i_traj,
                                                ref_filename=refPdb,
                                                bias_properties=bias_sproperties))

    ensemble = analysis.TrajectoryEnsemble(ref_filename=refPdb, 
                                           trajectories=trajectories)

    print('> Computing summary...')
    summary = ensemble.summary()
    info = open(f"{dirAn}/INFO.txt",'w')
    info.write(summary)
    info.close()

    print(summary)

    print('> Saving RMSD plots...')
    ensemble.plotRMSDs(f"{dirAn}/RMSD")

    print('> Saving average penalty plot...')
    try:
        ensemble.plotMeanPenalty(dirAn)
    except:
        print('> Trajectories have different lenghts, abort!')