import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mlp
import scipy.stats as stats
import os
import math
from string import ascii_uppercase
from matplotlib.patches import Rectangle

#NB Kb*T=kT = 4.11×10−21 J according to Wikipedia

def emptylists_matrix(dimension):
	"""
	makes matrix of empty lists
	I want a matrix (24x24) to keep bins consistent with previous 2D plots
	"""
	matrix_empty=np.empty((dimension,dimension), dtype=np.object_) #NB important to specify type of values otherwise numpy can't understand
	for i in range(matrix_empty.shape[0]):
		for j in range(matrix_empty.shape[1]):
			matrix_empty[i,j]=[]
	return(matrix_empty)

def probability_matrix(filename, dimension):
	"""
	Genrate probability matrix
	"""
	data = np.loadtxt(filename, delimiter=',', skiprows=1)
	Q = data[:,0]
	RMSD = data[:,1]
	max_RMSD = np.max(RMSD)
	frames = data[:,2].astype(int)
	id_traj = data[:,3].astype(int)
	matrix_counts = np.zeros((dimension,dimension))
	matrix_txt = emptylists_matrix((dimension))
	bin_size_Q = 1/dimension
	bin_size_RMSD= max_RMSD/dimension
	for i in frames:
		q_norm=int(Q[i]/bin_size_Q)
		rmsd_norm=int(RMSD[i]/bin_size_RMSD)
		if q_norm and rmsd_norm:	
			matrix_counts[q_norm-1][rmsd_norm-1] += 1
			matrix_txt[q_norm-1][rmsd_norm-1].append([id_traj[i],frames[i]])
	probability_mat=np.divide(matrix_counts, (len(frames))) 
	return (probability_mat, matrix_txt, bin_size_Q, bin_size_RMSD, max_RMSD)

def probability_plot(probability_mat, bin_size_Q, bin_size_RMSD, maximal_RMSD, filename):
	fig, ax = plt.subplots(figsize=(10, 10))
	ax.set_aspect('equal')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, maximal_RMSD])
	xlocs, xticks=plt.xticks(np.arange(0, 1.1, 0.1), rotation=90)
	xticks[0].set_visible(False) #removes tick at 0 so the plot is less cluttered
	ylocs, yticks=plt.yticks(np.arange(0, maximal_RMSD, 0.5))
	yticks[0].set_visible(False)
	plt.tick_params(axis = "both", which = "both", bottom = False, top = False, left=False, right=False)
	plt.xlabel('Q', fontsize=14)
	plt.ylabel('RMSD', fontsize=14)
	p=plt.imshow(probability_mat.T, origin = "lower", interpolation = "gaussian", extent=[0,1,0,maximal_RMSD], aspect=float(bin_size_Q/bin_size_RMSD), cmap='Blues') #set aspect to ratio of x unity/y unity to get square plot
	cbar=plt.colorbar(p, ax=ax, ticks=np.arange(0,np.amax(probability_mat), 0.02), shrink=0.805)
	cbar.set_label('Probability', fontsize=14, rotation=90)
	plt.savefig(filename)

def make_matrix_energy(probability_mat):
	"""
	Converts probability matrix into energy matrix
	energy = -log(probability)
	"""
	matrix_energy=np.where(probability_mat > 0, -np.log(probability_mat), 100)
	#rescale so that min value is 0
	matrix_energy_rescaled = np.where(matrix_energy!=100, matrix_energy-np.amin(matrix_energy), 100)
	real_values = np.where(matrix_energy_rescaled!=100, matrix_energy_rescaled, -100)
	matrix_energy_rescaled=np.where(matrix_energy!=100, matrix_energy_rescaled, np.amax(real_values)+1)
	return(matrix_energy_rescaled, real_values)

def energy_plot(matrix_energy_rescaled, bin_size_Q, bin_size_RMSD, maximal_RMSD, real_values, filename,squares=None):
	"""
	Plot the energy
	optionally plot the regions where you found intermediates
	"""
	fig, ax= plt.subplots(figsize=(10, 10))
	ax.set_title('-ln(p)')
	#plan graph...
	ax.set_aspect('equal')
	ax.set_xlim([0, 1])
	ax.set_ylim([0, maximal_RMSD])
	ax.autoscale(enable=True, axis='y',tight=True)
	xlocs, xticks=plt.xticks(np.arange(0, 1.1, 0.1), rotation=90)
	xticks[0].set_visible(False)
	ylocs, yticks=plt.yticks(np.arange(0, maximal_RMSD, 0.5))
	yticks[0].set_visible(False)
	plt.tick_params(axis = "both", which = "both", bottom = False, top = False, left=False, right=False)
	plt.xlabel('Q', fontsize=14)
	plt.ylabel('RMSD', fontsize=14)
	#generate the plot
	cmap = mlp.colormaps['RdYlBu'] #grab standard colormap
	cmap=cmap.reversed() #inverts color range to get blue for lower energy as convention!!
	cmap.set_over('white') #needed to set all values above threshold=unreal values to white
	p=plt.imshow(matrix_energy_rescaled.T, origin='lower', extent=[0,1,0.01,maximal_RMSD], interpolation='gaussian', aspect=float(bin_size_Q/bin_size_RMSD), cmap=cmap, vmax=np.amax(real_values)) #set aspect to ratio of x unity/y unity to get square plot
	#cbar=plt.colorbar(p, ax=ax, ticks=np.arange(np.amin(matrix_energy_rescaled.T), np.amax(real_values), 1.0), shrink=0.82)
	cbar=plt.colorbar(p, ax=ax, ticks=np.arange(1, np.amax(real_values), 1.0), shrink=0.805)
	plt.clim(2.5, np.amax(real_values)) #sets range for colorbar
	cbar.set_label('-ln(p)', fontsize=14, rotation=90)
	# Grid to help select ranges of Q and RMSD
	plt.grid()

	if squares != None:
		# Show regions where you extracted intermediates
		colours = dict(zip(squares, plt.cm.tab10.colors[:len(squares)]))
		labels = dict(zip(squares,[ascii_uppercase[i] for i in range(len(squares))]))
		for tupl in squares:
			ax.add_patch(Rectangle((tupl[0], tupl[2]), float(tupl[1]-tupl[0]), float(tupl[3]-tupl[2]), edgecolor=colours[tupl], facecolor="None" ,label=labels[tupl], linewidth=1.5 ))
		ax.legend(loc='upper right', framealpha=1, edgecolor='white')
	plt.savefig(filename)


#inputs:
size=25
selected_regions=[(0.72,0.95,0.05,0.37),(0.68,0.81,0.65,1.1),(0.43,0.61,1.5,2)] #list of tuples with minQ, maxQ, minRMSD, maxRMSD FROM CLOSEST TO NATIVE TO UNFOLDED!!!

#main:
# probability_plot(probability_matrix, Qbin, RMSDbin, max_RMSD)
# energy_matrix, real_values=make_matrix_energy(probability_matrix, max_RMSD, size)
# energy_plot(energy_matrix, Qbin, RMSDbin, max_RMSD, real_values, selected_regions)

filename = "banana.txt"

pic = "banana.png"

probability_matrix, allframes_matrix, Qbin, RMSDbin, max_RMSD =probability_matrix(filename, size)
probability_plot(probability_matrix, Qbin, RMSDbin, max_RMSD, filename="banana_probability.png")
energy_matrix, real_values=make_matrix_energy(probability_matrix, max_RMSD, size)
energy_plot(energy_matrix, Qbin, RMSDbin, max_RMSD, real_values,filename="babana_energy.png")