import pandas
import matplotlib.pyplot as plt
import scipy
import itertools
import scipy.interpolate

# Read in dataset
df = pandas.read_csv('hw5_prob2.csv')
xvals = df['x'].values
yvals = df['y'].values
zvals = df['log T'].values
num = len(zvals)

def plot_scatter():
	# Get min, mid, and max for colorbar ticks
	mn = scipy.amin(zvals)
	mx = scipy.amax(zvals)
	md = (mx+mn)/2

	# Plot dataset
	plt.scatter(xvals,yvals,c=zvals,s=800,cmap='Greys_r',vmin=mn,vmax=mx)
	plt.colorbar(ticks=[mn,md,mx])
	title = 'Jordan Aquifer Data'
	plt.title(title, y=1.04, fontsize=56)
	plt.xlabel('x',fontsize=56)
	plt.ylabel('y',fontsize=56)
	plt.rc('font', size=56)
	plt.tick_params(axis='both',labelsize=56)
	plt.grid()
	plt.show()

def variogram_data(div=15):
	# Square difference between z-values
	print num*(num-1)/2 # check number of pairs
	z_pairs = scipy.array(list(itertools.combinations(zvals,r=2)))
	z_sqdif = scipy.square( z_pairs[:,0] - z_pairs[:,1] )
	vg = pandas.DataFrame()
	vg['z_sqdif'] = z_sqdif

	# Square distance along x-axis
	x_pairs = scipy.array(list(itertools.combinations(xvals,r=2)))
	x_sqdif = scipy.square( x_pairs[:,0] - x_pairs[:,1] )
	# Square distance along y-axis
	y_pairs = scipy.array(list(itertools.combinations(yvals,r=2)))
	y_sqdif = scipy.square( y_pairs[:,0] - y_pairs[:,1] )
	# Linear distance between points
	xy_dist = scipy.sqrt( x_sqdif + y_sqdif )
	vg['xy_dist'] = xy_dist

	vg.to_csv('hw5_variogram_data.csv',index=False)

plot_scatter()
variogram_data()





