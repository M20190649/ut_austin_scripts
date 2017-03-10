# Paul J. Ruess
# University of Texas at Austin
# Fall, 2016

from netCDF4 import Dataset
import pandas
import scipy
import scipy.interpolate
from scipy.stats import norm
from collections import Counter
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class RCData:

	def __init__(self, comid, hand_props, 
		hand_props_idx, xs, idlookup):
		"""Provides hand, xs, and usgs rating curve data for the specified comid.
		'comid' - comid for which data is desired
		'hand_curves' - NetCDF file containing HAND rating curve data
		'hand_curves_idx' - csv containing indices of HAND rating curves desired
		'hand_props' - NetCDF file containing HAND hydraulic property data
		'hand_props_idx' - csv containing indices of HAND hydraulic properties desired
		'xs' - csv containing xs data (must have profile, xsid, and rating curves)
		'idlookup' - csv lookup table between comid and usgsid"""
		self.comid = comid
		
		self.hand_props = Dataset(hand_props,'r')
		hand_props_idx = pandas.read_csv(hand_props_idx)
		self.hand_props_idx = hand_props_idx.loc[ hand_props_idx['comid'] == \
			self.comid ]['index'].values[0]
		self.get_hand_props()

		self.idlookup = idlookup
		
		self.xs = xs[ xs['COMID'] == str(comid)]
		if self.get_xs() == None: # Fetch xs cross-sections for this comid
			raise TypeError

	def get_hand_props(self):
		"""Initializes self.handarea [sqmeters], self.handrad [m], 
		self.handslope [-], and self.handstage [ft]."""
		handc = self.hand_props.variables['COMID']
		handslope = self.hand_props.variables['Slope'] # So
		handstage = self.hand_props.variables['StageHeight'] # h values for Aw and Hr
		handarea = self.hand_props.variables['WetArea'] # Aw
		handrad = self.hand_props.variables['HydraulicRadius'] # Hr
		handlen = self.hand_props.variables['Length'] # Length
		if handc[self.hand_props_idx] == self.comid:
			self.handarea = handarea[self.hand_props_idx]*10.7639 # Convert sqm to sqft
			self.handrad = handrad[self.hand_props_idx]*3.28084 # Convert m to ft
			self.handslope = handslope[self.hand_props_idx] # unitless
			self.handlen = handlen[self.hand_props_idx]*3.28084 # Convert m to ft
		handstage = scipy.array(handstage)*3.28084 # Convert m to ft
		self.handstage = scipy.rint(handstage) # Round to nearest int

	def get_xs(self):
		""" Retrieve xs information to populate self.xscurves. 
		Also create and populate self.max_q_query with maximum q 
		value for querying interpolated rating curves."""
		prof = []
		disch = []
		stage = []
		self.max_q_query = 0
		self.max_disch = 0
		self.max_h_query = 0
		self.max_stage = 0

		# Retrieve xs information and populate self.xscurves
		stations = self.xs['RiverStation'].unique()
		# a = self.xs[self.xs['RiverStation'].isin(stations)]
		for i, rs in enumerate( stations ): 
			# stage-height values for RiverStation rs
			h = self.xs[self.xs['RiverStation'] == rs]['Stage_Height_ft_'].values
# ************
# If multiple zeros, ignore this RiverStation and proceed to next
# ************
			# Test if repeated zeroes (ie. multiple xs datasets for this RiverStation)
			repeats = [item for item, count in Counter(h).iteritems() if count > 1]
			if repeats: continue
			# Process xs data
			current = self.xs[ self.xs['RiverStation'] == rs ]
			prof.append(current['ProfileM'].unique()[0]) # xs location along reach
			disch.append(map(float,current['Discharge_cfs_'].values)) # disch vals
			stage.append(map(float,current['Stage_Height_ft_'].values)) # stage vals

			# Find max q value for querying interpolations
			# Find max disch value for plotting x_axis
			max_disch = int( scipy.amax(disch[-1]) )
			if self.max_q_query == 0:
				self.max_q_query = max_disch
				self.max_disch = max_disch	
			elif max_disch < self.max_q_query: 
				self.max_q_query = max_disch
			elif max_disch > self.max_disch: 
				self.max_disch = max_disch

			# Find max h value for querying interpolations
			# Find max stage value for plotting y_axis
			max_stage = int( scipy.amax(stage[-1]) )
			if self.max_h_query == 0:
				self.max_h_query = max_stage
				self.max_stage = max_stage	
			elif max_stage < self.max_h_query: 
				self.max_h_query = max_stage
			elif max_stage > self.max_stage: 
				self.max_stage = max_stage
		if len(disch) != 0:
			xs_profs = scipy.array(prof).astype(float)
			self.xs_profs = scipy.unique(xs_profs) # remove repeats
			self.xs_disch = scipy.array(disch)
			self.xs_stage = scipy.array(stage)
			# print '\n------------------------\n'
			# for s,p in zip(stations,self.xs_profs): print s,p
			# print '\n------------------------\n'
			return 1

class RCDist(RCData):

	def interp(self,x,y,kind='power'):
		"""Interpolate over data with (x,y) pairs
		'x' - x data,
		'y' - y data,
		'kind' - powerlaw ('power'), linear ('linear), or cubic ('cubic')"""
		if kind == 'power': # powerlaw interpolation
			logx = scipy.log(x)[1:] # ln(0) is neg inf, so remove first term
			logy = scipy.log(y)[1:] # ln(0) is neg inf, so remove first term
			b, loga = scipy.polyfit(logx,logy,1) # slope, intercept from (y = a + b*x)
			a = scipy.exp(loga)
			f = lambda x: a * x**b # powerlaw function

		if kind == 'linear' or kind == 'cubic': # linear interpolation
			f = scipy.interpolate.interp1d(x,y,kind=kind)

		return f

	def get_disch(self,h,kind='power'):
		"""Given stage-height h, query interpolated rating curves
		and return associated discharges."""
		other_dischs = []
		for disch,stage in zip(self.xs_disch,self.xs_stage):
			f = self.interp(x=stage,y=disch,kind=kind) # interpolation function
			other_dischs.append( f(h) )
		return (scipy.array(other_dischs).T)

	def multi_boxplot(self,upto=83):
		fig, ax = plt.subplots()
		fig.set_size_inches(20,16, forward=True)

		curdischs = self.get_disch(self.handstage)[:upto]
		ax.boxplot(curdischs.T, 0, sym='rs', vert=0)

		yticks = scipy.arange(0,len(self.handstage[:upto]),10)
		ax.set_yticklabels(yticks,rotation=0)
		ax.yaxis.set_ticks(yticks)
		title = 'COMID {0} Rating Curve Boxplots'.format(self.comid)
		ax.set_title(title, y=1.04, fontsize=56)
		plt.xlabel('Q (cfs)',fontsize=56)
		plt.ylabel('H (ft)',fontsize=56)
		ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
		plt.rc('font', size=56)
		# plt.legend(loc='upper left',fontsize=40)
		# plt.tick_params(axis='both',labelsize=56)
		plt.grid()

		ax.tick_params(labelsize=56)
		# fig.savefig('results/stochhydro/multi_boxplot_{0}.png'.format(str(self.comid)))
		plt.show()

	def mannings_n(self,area,hydrad,slope,disch):
		""" Calculates manning's roughness from discharge. 
		'area' - self.handarea (wet area),
		'hydrad' - self.handrad (hydraulic radius),
		'slope' - self.handslope (bed slope), and
		'disch' - any discharge values"""
		res = 1.49*area*scipy.power(hydrad,(2/3.0))*scipy.sqrt(slope)/disch.T
		return res.T

	def get_xs_n(self,low,upto):
		"""Returns an array containing n-values for each cross-section
		at each 1ft stage-height interval.
		'upto' - describes how many stage-heights to iterate through
			when collecting n-values for all cross-sections"""
		area = self.handarea[low:upto]
		hydrad = self.handrad[low:upto]
		slope = self.handslope
		disch = self.get_disch(h=self.handstage[low:upto],kind='power') # xs values only
		xs_nvals = self.mannings_n(area=area,hydrad=hydrad,slope=slope,disch=disch)
		self.xs_nvals = scipy.nan_to_num( xs_nvals )
		return self.xs_nvals

	def nstats(self,low,upto):
		"""Given an n-values array returns the average and standard deviation
		of the n-values taken at each stage-height for each cross-section.
		'nvals' - array of n-values with shape (x,y) with cross-sections
			along the x axis and stage-heights along the y axis"""
		self.get_xs_n(low=low,upto=upto)
		means = scipy.mean(self.xs_nvals,axis=1)
		stdevs = scipy.std(self.xs_nvals,axis=1)
		# if filename = True: # Write to csv
		return scipy.column_stack((self.handstage[low:upto],means,stdevs))

	def mannings_q(self,area,hydrad,slope,n):
		""" Calculates manning's discharge from roughness. 
		'area' - self.handarea (wet area),
		'hydrad' - self.handrad (hydraulic radius),
		'slope' - self.handslope (bed slope), and
		'n' - any roughness values"""
		res = 1.49*area*scipy.power(hydrad,(2/3.0))*scipy.sqrt(slope)/n.T
		return res.T

	def get_xs_q(self,low,upto):
		"""Returns an array containing q-values for each cross-section
		at each 1ft stage-height interval, calculated from average n-values.
		'upto' - describes how many stage-heights to iterate through
			when collecting n-values for all cross-sections"""
		area = self.handarea[low:upto]
		hydrad = self.handrad[low:upto]
		slope = self.handslope
		n = self.nstats(low=low,upto=upto)[:,1] # xs mean n-values
		xs_qvals = self.mannings_q(area=area,hydrad=hydrad,slope=slope,n=n)
		self.xs_qvals = scipy.nan_to_num( xs_qvals )
		return (self.xs_qvals,self.handstage[low:upto])

	def plot_rc(self,save=False,xs=True,xsapprox=True,
		kind='power',dist=5000,raw=False,alpha=0.05,div=5,
		box=False):
		"""Plot HAND and xs rating curves with confidence intervals
		'hand' - plot hand rating curve [T/F]
		'xs' - plot xs rating curves [T/F]
		'xsapprox' - plot xs rating curve approximation from n-value averages [T/F]
		'ci' - plot confidence intervals [T/F]
		'alpha' - alpha for confidence intervals [float(0.0,1.0)]
		'div' - number of intervals for confidence interval [R]"""
		
		if xs: # Plot all linearly-interpolated XS rating curves
			intervals = scipy.arange(dist,self.handlen+dist,dist)
			# print 'Intervals:',intervals

			cutoffub = [i/self.handlen*100 for i in intervals]
			cutofflb = scipy.copy(cutoffub)
			cutofflb = scipy.insert(cutofflb,0,0)[:-1]
			cutoffs = zip(cutofflb,cutoffub)
			for l,u in cutoffs:
				idx = scipy.where(scipy.logical_and(
					scipy.greater_equal(self.xs_profs,l),
					scipy.less(self.xs_profs,u)))[0]
				if u > 100: u = 100.00

				fig, ax = plt.subplots() # get figure and axes for plotting
				fname = 'results/by5000/{0}/rc__comid_{0}_from_{1}_to_{2}.png'.format(
					self.comid,('%.2f' % l),('%.2f' % u))

				for prof,disch,stage in zip(self.xs_profs[idx],self.xs_disch[idx],self.xs_stage[idx]):
					# Get interpolation function
					# print (('%.2f' % prof) + str(disch))
					# print (('%.2f' % prof) + str(stage))

					f = self.interp(x=disch,y=stage,kind=kind)

					if raw == True: # Plot raw data (ie. only HEC-RAS points)
						# interp over discharge
						ax.plot(disch,f(disch),c='grey', linewidth=2)

						# interp over stage (switched axes) for testing
						# f = self.interp(x=stage,y=disch,kind=kind)
						# ax.plot(f(stage),stage,c='purple',linewidth=1)

					if raw == False: # Plot interpolated data (ie. 'div' many interpolated points)
						interval = disch[-1] / div
						qvals = scipy.arange( 0, (disch[-1]+interval), interval ) # [1:]
						ax.plot(qvals,f(qvals),c='grey',linewidth=2)

				# Add one label for all cross-section curves		
				ax.plot([],[],label='HEC-RAS',c='grey',linewidth=2)
				# Plot graph
				fig.set_size_inches(20,16, forward=True)
				plt.gca().set_xlim(left=0,right=self.max_disch)
				plt.gca().set_ylim(bottom=0,top=self.max_stage)
				ax.set_xticks(ax.get_xticks()[::2])
				ax.set_yticks(ax.get_yticks()[::2])
				title = 'COMID {0}, ({1},{2})'.format(self.comid,('%.2f' % l),('%.2f' % u))
				ax.set_title(title, y=1.04, fontsize=56)
				plt.xlabel('Q (cfs)',fontsize=56)
				plt.ylabel('H (ft)',fontsize=56)
				ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
				plt.rc('font', size=56)
				plt.legend(loc='upper left',fontsize=40)
				plt.tick_params(axis='both',labelsize=56)
				plt.grid()	

				# print '\n------------------------\n'
				if xsapprox:
					# Add approximate rating curve from average n-values
					qvals,hvals = self.get_xs_q(low=0,upto=83)
					f = self.interp(x=qvals,y=hvals,kind=kind)
					ax.plot(qvals,f(qvals),label='Resistance Function',c='red',linewidth=5)

					# Add approximate rating curve for these indices
					idxqvals,idxhvals = self.get_xs_q(low=idx[0],upto=idx[-1])

					if len(idxqvals) == 0:
						print 'No data found for profiles {0} to {1}'.format(('%.2f' % l),('%.2f' % u))
						break

					f = self.interp(x=idxqvals,y=idxhvals,kind=kind)
					ax.plot(idxqvals,f(idxqvals),label='Resistance Function Local Average',c='orange',linewidth=5)			

		else: fig,ax = plt.subplots()

		if save: fig.savefig(fname)
		else: 
			# mng = plt.get_current_fig_manager()
			# mng.resize(*mng.window.maxsize())
			plt.show()

if __name__ == '__main__':
	
	# Path to HAND files
	hand_props = 'oniondata/OnionCreek.nc'
	hand_props_idx = 'oniondata/handnc_idx.csv'

	# Pre-process XS data
	xsintersect = pandas.read_csv('oniondata/xsdata/xsintersect.csv',
		usecols=['COMID','ProfileM','RiverStation'])
	xsintersect = xsintersect.astype(str)
	xsrating = pandas.read_csv('oniondata/xsdata/xsrating.csv',
		usecols=['Stage_Height_ft_','Discharge_cfs_','RiverStation','XSID'])
	xsrating = xsrating.astype(str)
	xs = pandas.merge(xsrating,xsintersect,on='RiverStation')

	# Pre-process id lookup table (USGSID <--> COMID)	
	idlookup = pandas.read_csv('oniondata/streamgages.csv',usecols=['SOURCE_FEA','FLComID'])
	idlookup['SOURCE_FEA'] = idlookup['SOURCE_FEA'].apply(lambda x: '{0:0>8}'.format(x))
	
	# Get lists of COMIDs to iterate over
	comids = idlookup.FLComID.unique()

	# Override with reasonable-looking comids (from multi-boxplots)
	comids = [5781431,5781477,5781369]

	# Instantiate RCDist class for each comid in watershed
	for comid in comids:
		try: 
			rcdist = RCDist(comid,hand_props,hand_props_idx,xs,idlookup)
			print 'COMID {0} Collected Suffessfully!'.format(str(comid))
		except TypeError: 
			print 'COMID {0} XS Error'.format(str(comid))
			continue
		except IndexError: 
			print 'COMID {0} RC Error'.format(str(comid))
			continue

		# Retrieve nstats
		# rcdist.multi_boxplot()

		# Plot rating curves from data
		rcdist.plot_rc(save=False,xs=True,xsapprox=True,dist=2000,
			raw=True,kind='linear',alpha=0.05,div=5)