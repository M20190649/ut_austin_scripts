#title           :hw1.py
#description     :HW 1 for ORI 397
#author          :Paul J. Ruess
#date            :20160826
#==============================================================================

import scipy as sp
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Function for creating plots using specified inputs
def plot_dist_grid(length=20,start=[(0,0)],n=1,metric='euclidean',interp=None,cmap=None,cbar=False,title='Distance to Closest Point',xlabel="X Coord",ylabel="Y Coord"): 

	# Create array of x and y coordinates
	x_array = sp.zeros((length,length)) + sp.arange(length)
	y_array = sp.zeros((length,length)) + sp.expand_dims(sp.arange(length),length)
	coords = zip(x_array.ravel(),y_array.ravel())

	# Iterate over coords to calculate distance from 'start'
	minima = []
	for i in range(len(start)):
		val = distance.cdist([start[i]], coords, metric).reshape(length,length)
		if i == 0:
			minima = sp.copy(val) # Assume all are minimums
		else: 
			minima = sp.minimum(minima,val) # Take smaller

	# Create plot from 'minima' array
	fig, ax = plt.subplots()
	cax = ax.imshow(minima,interpolation=interp,cmap=cmap)
	if cbar: 
		cbar = fig.colorbar(cax, ticks=[range(int(sp.amax(minima)))])
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	fig.savefig('figure' + str(n) + '.pdf')
	# plt.show()

# Request seven plots for homework assignment (in order)
plot_dist_grid()
plot_dist_grid(start=[(0,0)],n=2,interp='nearest')
plot_dist_grid(start=[(5,5)],n=3,interp='nearest')
plot_dist_grid(start=[(5,5)],n=4,interp='spline36',cmap='hot_r')
plot_dist_grid(start=sp.array([(0,0),(5,5),(19,19),(19,0),(0,19)]),n=5,interp='spline36',cmap='hot_r')
plot_dist_grid(start=sp.array([(0,0),(5,5),(19,19),(19,0),(0,19)]),n=6,interp='nearest',cmap='hot_r')
plot_dist_grid(start=sp.array([(0,0),(5,5),(19,19),(19,0),(0,19)]),n=7,cbar=True, interp='nearest',cmap='hot_r')