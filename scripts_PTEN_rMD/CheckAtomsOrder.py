import os

"""
Check if the order of aroms is the same in the initial and final topology
"""

pdb = "/home/annarita.zanon/ratchet/PTEN/conf/PTEN.pdb"
pre = "/home/annarita.zanon/ratchet/PTEN/data/iter_2/unfolding/init_conf.gro"
post ="/home/annarita.zanon/ratchet/PTEN/data/iter_2/ratchet_1/ratchet_md.gro"

pdb = "/home/annarita.zanon/ratchet/PTEN_Y178D/conf/PTEN_Y178D.pdb"
pre = "/home/annarita.zanon/ratchet/PTEN_Y178D/conf/unfolding_init/init_conf.gro"
post ="/home/annarita.zanon/ratchet/PTEN_Y178D/data/iter_1/ratchet_1/nvt.gro"


with open(pdb, 'r') as f:
    pdb_lines = [line.strip().split() for line in f.readlines() if line.startswith("ATOM")]
    pdb_lines = [[f"{line[5]}{line[3]}", line[2], line[1]] for line in pdb_lines]

# print(pdb_lines[1])
# print(pdb_lines[-1])

with open(pre, 'r') as f:
    pre_lines = [line.strip().split() for line in f.readlines()][2:-1]
    pre_lines = [[line[0], line[1], line[2]] for line in pre_lines]

# print(pre_lines[1])
# print(pre_lines[-1])

with open(post, 'r') as f:
    post_lines = [line.strip().split() for line in f.readlines()][2:len(pre_lines)+2]
    post_lines = [[line[0], line[1], line[2]] for line in post_lines]

# print(post_lines[1])
# print(post_lines[-1])

print("\t PRE vs. POST")
for i in range(len(pdb_lines)):
    pdb_line = pdb_lines[i]
    pre_line = pre_lines[i]
    post_line = post_lines[i]
    pdb_atomtype = pdb_line[1][0]
    pre_atomtype = pre_line[1][0]
    post_atomtype = post_line[1][0]
    if pre_atomtype != post_atomtype:
        print(f"pre: {pre_line} post: {post_line}")

print("\t PRE vs. PDB")
for i in range(len(pdb_lines)):
    pdb_line = pdb_lines[i]
    pre_line = pre_lines[i]
    post_line = post_lines[i]
    pdb_atomtype = pdb_line[1][0]
    pre_atomtype = pre_line[1][0]
    post_atomtype = post_line[1][0]
    if pdb_atomtype != pre_atomtype:
        print(f"pre: {pre_line} pdb: {pdb_line}")

print("\t POST vs. PDB")
for i in range(len(pdb_lines)):
    pdb_line = pdb_lines[i]
    pre_line = pre_lines[i]
    post_line = post_lines[i]
    pdb_atomtype = pdb_line[1][0]
    pre_atomtype = pre_line[1][0]
    post_atomtype = post_line[1][0]
    if pdb_atomtype != post_atomtype:
        print(f"post: {post_line} pdb: {pdb_line}")