import numpy as np
from FoldingAnalysis.analysis import *
import re
import json
import FoldingAnalysis as fa
import FoldingAnalysis.utilities as utilities
import numpy as np

sysName = "PTEN_WT"

targetConf = f"/home/annarita.zanon/ratchet/{sysName}/conf/{sysName}.pdb"
Cmap = Structure(targetConf).sarr
print(Cmap)
# Cmap.astype(np.uint8)

# len_cmap = Cmap.shape[0]

# with open(targetConf, 'r') as f:
#     resid = []
#     for line in f.readlines():
#         if line.split(" ")[0] == "ATOM":
#             splitted = re.split(" +", line)
#             resid.append(int(splitted[5]))
# len_pdb = max(resid) - min(resid) + 1

# if len_pdb == len_cmap:
#     print(f"Protein and contact map have the same dimension of {len_pdb}")
# else:
#     print(f"Warning!\nProtein has {len_pdb} residues, map has dimension {len_cmap}")

native = fa.Trajectory('/home/annarita.zanon/ratchet/PTEN_WT/conf/PTEN_WT.pdb')

sign = native.getCmapSignature()
print(sign)

cmap = native.getCmap()
print(cmap)

utilities.saveCmapNew('native.cmp', cmap, sign, 1/(np.linalg.norm(cmap)**2), 1)
