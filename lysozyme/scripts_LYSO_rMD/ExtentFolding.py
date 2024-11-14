"""
Extends a simulation of x picoseconds
"""

import os
import multiprocessing


num_ratchet = 1
iteration = 1
x = 6000
working_dir = f"/home/annarita.zanon/ratchet/lysozyme/LYSO/data/iter_{iteration}/ratchet_{num_ratchet}"

gmx = "gmx"
numThreads = multiprocessing.cpu_count()


os.chdir(working_dir)

os.system(f"{gmx} convert-tpr -s ratchet_md.tpr -extend {x} -o ratchet_md.tpr")

os.system(
    f"{gmx} mdrun -deffnm ratchet_md -s ratchet_md -cpi ratchet_md.cpt -noappend -bonded gpu -nb gpu -pmefft gpu -pme gpu -nt {int(numThreads)}"
    )

os.system(
        f"{gmx} trjconv -s ratchet_md.tpr -f ratchet_md.xtc -ur compact -center -pbc mol -o ratchet_md_noPBC.xtc << EOF \n1\n1\nEOF"
    )