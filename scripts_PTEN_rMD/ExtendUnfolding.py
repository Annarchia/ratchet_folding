"""
Extends a simulation of x picoseconds
"""

import os
import multiprocessing

working_dir = "/home/annarita.zanon/ratchet/PTEN/data/iter_1/unfolding"

gmx = "gmx"
x = 2000
numThreads = multiprocessing.cpu_count()


os.chdir(working_dir)

os.system(f"{gmx} convert-tpr -s unfolding_md.tpr -extend {x} -o unfolding_md_ext.tpr")

os.system(
    f"{gmx} mdrun -s unfolding_md_ext -cpi state.cpt -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
    )

os.system(
        f"{gmx} trjconv -s unfolding_md_ext.tpr -f unfolding_md_ext.xtc -ur compact -center -pbc mol -o unfolding_md_ext_noPBC.xtc << EOF \n1\n1\nEOF"
    )