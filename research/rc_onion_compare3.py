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

class RCData:

	def __init__(self, comid, hand_curves, hand_curves_idx, hand_props, 
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
		
		self.hand_curves = Dataset(hand_curves,'r')
		hand_curves_idx = pandas.read_csv(hand_curves_idx)
		self.hand_curves_idx = hand_curves_idx.loc[ hand_curves_idx['comid'] == \
			self.comid ]['index'].values[0]
		self.get_hand_curves()
		
		self.hand_props = Dataset(hand_props,'r')
		hand_props_idx = pandas.read_csv(hand_props_idx)
		self.hand_props_idx = hand_props_idx.loc[ hand_props_idx['comid'] == \
			self.comid ]['index'].values[0]
		self.get_hand_props()

		self.idlookup = idlookup
		self.usgsids = self.idlookup.loc[self.idlookup['FLComID'] == \
			self.comid]['SOURCE_FEA'].values
		self.usgsrc = 'http://waterdata.usgs.gov/nwisweb/get_ratings?' \
			'file_type=exsa&site_no={0}'
		
		if self.get_usgsrc() == 0: # Fetch usgs stage and disch values
			raise IndexError
		
		self.xs = xs[ xs['COMID'] == str(comid)]
		if self.get_xs() == None: # Fetch xs cross-sections for this comid
			raise TypeError

	def get_hand_curves(self): 
		""" Initializes self.handq [cfs] and self.handh [ft]."""
		handq = self.hand_curves.variables['Q_cfs']
		handh = self.hand_curves.variables['H_ft']
		handc = self.hand_curves.variables['COMID']
		if handc[self.hand_curves_idx] == self.comid:
			self.handq = handq[self.hand_curves_idx]
		self.handh = handh

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

	def get_usgsrc(self):
		""" Initializes self.usgsq and self.usgsh """
		import urllib
		import re
		self.usgsh = []
		self.usgsq = []
		for usgsid in self.usgsids:
			urlfile = urllib.urlopen(self.usgsrc.format(str(usgsid)))
			urllines = urlfile.readlines()
			findData = False
			usgsq = scipy.array([])
			usgsh = scipy.array([])
			for j in range(len(urllines)):
				line = urllines[j]
				if not findData and not re.search('[a-zA-Z]',line): # No letters
					findData = True
				if findData and float(line.split('\t')[2]) >= 1: # Remove where Q < 1
					current = line.split('\t')
					usgsq = scipy.append( usgsq, float(current[2]) )
					# apply shift to stage height where current[1] is shift magnitude
					usgsh = scipy.append( usgsh, float(current[0]) - float(current[1]) )
			shift = usgsh[0]
			self.usgsh.append((usgsh - shift)) # Normalize usgsh over bottom depth
			self.usgsq.append(usgsq)
		self.usgsh = scipy.array(self.usgsh)
		self.usgsq = scipy.array(self.usgsq)

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
			# Test if repeated zeroes (ie. multiple xs datasets for this RiverStation)
# ************
# If multiple zeros, ignore this RiverStation and proceed to next
# ************
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
			self.xs_profs = scipy.array(prof)
			self.xs_disch = scipy.array(disch)
			self.xs_stage = scipy.array(stage)
			return 1

class MapFlood(RCData):
	"""Class for retrieveing national water model forecasts and retrieving
	various stage-heights (from numerous rating curves in RCData) to create
	probabilistic stage-height distributions and plot probabilistic flood extents."""

	def nwm_stage(self,forecast):
		from pynwm import nwm
		"""Retrieve nwm forecast using pynwm module, and retrieve rating curves
		for self.comid, and provide probabilistic stage height distribution.
		'forecast' - type of forecast from nwm models, options being 'short_range', 
		'med_range', or 'analysis_assym'; 'long_range' is not included"""
		series = nwm.get_streamflow(forecast, self.comid, timezone='US/Central')
		interp = scipy.interpolate.interp1d(self.handq,self.handh) # kind=cubic

		self.qdict = {}
		if len(series) > 1: # long_range ensemble
			for s in series:
				datevals = zip( s['dates'], s['values'] )
				for (d, v) in (datevals):
					d = d.strftime('%y-%m-%d %H')
					self.qdict[d] = v
		else: # all other models (short_range, med_range, analysis_assym)
			s = series[0]
			datevals = zip( s['dates'], s['values'] )
			for (d, v) in (datevals):
				d = d.strftime('%y-%m-%d %H')
				self.qdict[d] = v

		fcast_d = scipy.array(self.qdict.keys())
		fcast_v = scipy.array(self.qdict.values())
		return ( fcast_d, interp(fcast_v) )

	def map_flood(self):
		"""Map probabilistic floods for specific stream reach based on various
		stage-heights retrieved from RCData rating curves."""
		pass

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

	def get_stage(self,q,kind='power'):
		"""Given discharge q, query interpolated rating curves
		and return associated stage-heights."""
		other_stages = []
		for disch,stage in zip(self.xs_disch,self.xs_stage):
			f = self.interp(x=disch,y=stage,kind=kind) # interpolation function
			other_stages.append( f(q) )

		# Interpolate over HAND
		f_hand = self.interp(x=self.handq,y=self.handh,kind=kind)
		hand_stage = f_hand(disch)
		return (hand_stage, other_stages)

	def get_disch(self,h,kind='power'):
		"""Given stage-height h, query interpolated rating curves
		and return associated discharges."""
		other_dischs = []
		for disch,stage in zip(self.xs_disch,self.xs_stage):
			f = self.interp(x=stage,y=disch,kind=kind) # interpolation function
			other_dischs.append( f(h) )

		# Interpolate over HAND
		f_hand = self.interp(x=self.handq,y=self.handh,kind=kind)
		hand_disch = f_hand(stage)
		return (scipy.array(hand_disch), scipy.array(other_dischs).T)

	def get_ci(self,alpha=0.05,div=5,axis='x'):
		"""Get upper- and lower-bounds (confidence interval)
		of normal distribution from xs stage_heights at 'div'
		evenly spaced intervals.
		'axis' - designates which axis to create bounds over, in this case
			'x' corresponds to discharge and 'y' corresponds to stage-height
		'alpha' - statistical alpha value
		'div' - number of intervals"""

		ubounds = [] # upper-bounds
		lbounds = [] # lower-bounds

		def get_bounds(mean,stdev,alpha):
			z = norm.ppf( (1-alpha/2), scale=stdev ) # z, mean = 0, 2-sided
			ub = mean+z # upper bound
			lb = mean-z # lower bound
			return ub,lb

		if axis == 'x':
			interval = self.max_q_query / div
			ci_qvals = scipy.arange( 0, (self.max_q_query+interval), interval ) # [1:]
			for q in ci_qvals: 
				hand_stage = self.get_stage(q)[0] # hand stage
				other_stages = self.get_stage(q)[1] # xs stages
				mean = scipy.average(other_stages) # mean of xs stages
				stdev = scipy.std(other_stages) # stdev of xs stages
				ub,lb = get_bounds(mean=mean,stdev=stdev,alpha=alpha)
				ubounds.append(ub)
				lbounds.append(lb)
			self.ci_vals = ci_qvals
		
		elif axis == 'y':
			interval = self.max_h_query / div
			ci_hvals = scipy.arange( 0, (self.max_h_query+interval), interval ) # [1:]
			for h in ci_hvals: 
				hand_disch = self.get_disch(h)[0] # hand disch
				other_dischs = self.get_disch(h)[1] # xs dischs		
				mean = scipy.average(other_dischs) # mean of xs dischs
				stdev = scipy.std(other_dischs) # stdev of xs dischs
				ub,lb = get_bounds(mean=mean,stdev=stdev,alpha=alpha)
				ubounds.append(ub)
				lbounds.append(lb)
			self.ci_vals = ci_hvals
		
		self.ubounds = scipy.nan_to_num( ubounds )
		self.lbounds = scipy.nan_to_num( lbounds )

	def mannings_n(self,area,hydrad,slope,disch):
		""" Calculates manning's roughness from discharge. 
		'area' - self.handarea (wet area),
		'hydrad' - self.handrad (hydraulic radius),
		'slope' - self.handslope (bed slope), and
		'disch' - any discharge values"""
		res = 1.49*area*scipy.power(hydrad,(2/3.0))*scipy.sqrt(slope)/disch.T
		return res.T

	def get_xs_n(self,upto=40):
		"""Returns an array containing n-values for each cross-section
		at each 1ft stage-height interval.
		'upto' - describes how many stage-heights to iterate through
			when collecting n-values for all cross-sections"""
		disch = []
		area = self.handarea[:upto]
		hydrad = self.handrad[:upto]
		slope = self.handslope
		disch = self.get_disch(h=self.handstage[:upto],kind='power')[1] # xs values only
		xs_nvals = self.mannings_n(area=area,hydrad=hydrad,slope=slope,disch=disch)
		self.xs_nvals = scipy.nan_to_num( xs_nvals )
		return self.xs_nvals

	def nstats(self,filename=False,upto=40):
		"""Given an n-values array returns the average and standard deviation
		of the n-values taken at each stage-height for each cross-section.
		'nvals' - array of n-values with shape (x,y) with cross-sections
			along the x axis and stage-heights along the y axis"""
		self.get_xs_n(upto)
		means = scipy.mean(self.xs_nvals,axis=1)
		stdevs = scipy.std(self.xs_nvals,axis=1)
		# if filename = True: # Write to csv
		return scipy.column_stack((self.handstage[:upto],means,stdevs))

	def mannings_q(self,area,hydrad,slope,n):
		""" Calculates manning's discharge from roughness. 
		'area' - self.handarea (wet area),
		'hydrad' - self.handrad (hydraulic radius),
		'slope' - self.handslope (bed slope), and
		'n' - any roughness values"""
		res = 1.49*area*scipy.power(hydrad,(2/3.0))*scipy.sqrt(slope)/n.T
		return res.T

	def get_xs_q(self,upto=83):
		"""Returns an array containing q-values for each cross-section
		at each 1ft stage-height interval, calculated from average n-values.
		'upto' - describes how many stage-heights to iterate through
			when collecting n-values for all cross-sections"""
		disch = []
		area = self.handarea[:upto]
		hydrad = self.handrad[:upto]
		slope = self.handslope
		n = self.nstats(upto=upto)[:,1] # xs mean n-values
		xs_qvals = self.mannings_q(area=area,hydrad=hydrad,slope=slope,n=n)
		self.xs_qvals = scipy.nan_to_num( xs_qvals )
		return (self.xs_qvals,self.handstage[:upto])

	def plot_rc(self,filename=False,hand=True,usgs=True,xs=True,xsapprox=True,
		ci=True,kind='power',raw=False,alpha=0.05,div=5):
		"""Plot HAND and xs rating curves with confidence intervals
		'hand' - plot hand rating curve [T/F]
		'xs' - plot xs rating curves [T/F]
		'xsapprox' - plot xs rating curve approximation from n-value averages [T/F]
		'ci' - plot confidence intervals [T/F]
		'alpha' - alpha for confidence intervals [float(0.0,1.0)]
		'div' - number of intervals for confidence interval [R]"""
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots() # get figure and axes for plotting

# # # # # TEST PLOT. Should be later removed. # # # # #
		# [84] returns very high n value (476) for first comid
		# f = self.interp(x=self.xs_disch[84],y=self.xs_stage[84],kind=kind)
		# interval = self.xs_disch[84][-1] / div
		# qvals = scipy.arange( 0, (self.xs_disch[84][-1]+interval), interval ) # [1:]
		# ax.plot(qvals,f(qvals),c='grey', linewidth=2)
		# ax.plot(self.xs_disch[84],f(self.xs_disch[84]),
			# label='Q-inf Example',c='purple', linewidth=6)

		if xs: # Plot all linearly-interpolated XS rating curves
			for disch,stage in zip(self.xs_disch,self.xs_stage):
				# Get interpolation function
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

		if xsapprox:
			# Add approximate rating curve from average n-values
			qvals,hvals = self.get_xs_q()
			f = self.interp(x=qvals,y=hvals,kind=kind)
			ax.plot(qvals,f(qvals),label='Resistance Function',c='red',linewidth=5)

		if ci: # Plot confidence interval bounds
			axis = 'x' # manually set to x to avoid problems with y plotting
			self.get_ci(alpha=alpha,div=div,axis=axis) # get upper- and lower-bound variables
			if axis == 'x':
				# upper bounds
				f = self.interp(x=self.ci_vals,y=self.ubounds,kind=kind)
				ax.plot(self.ci_vals,f(self.ci_vals),
					label='{0}%CI'.format( int((1-alpha)*100) ),c='orange', linewidth=5)
				# lower bounds
				f = self.interp(x=self.ci_vals,y=self.lbounds,kind=kind)
				ax.plot(self.ci_vals,f(self.ci_vals),
					label='{0}%CI'.format( int((1-alpha)*100) ),c='orange', linewidth=5)
# ************
# Not plotting properly
# ************
			if axis == 'y': 
				# upper bounds
				f = self.interp(x=self.ubounds,y=self.ci_vals,kind=kind)
				ax.plot( self.ci_vals, f(self.ci_vals),
					label='{0}%CI'.format( int((1-alpha)*100) ),c='orange', linewidth=5)
				# lower bounds
				f = self.interp(x=self.lbounds,y=self.ci_vals,kind=kind)
				ax.plot( self.ci_vals, f(self.ci_vals),
					label='{0}%CI'.format( int((1-alpha)*100) ),c='orange', linewidth=5)

		if usgs: # Plot interpolated USGS rating curve
			for q,h in zip(self.usgsq,self.usgsh):
				if kind == 'cubic': 
					print 'USGS interpolation plotted as power-law fit'
					f = self.interp(x=q,y=h,kind='power')
				else: 
					f = self.interp(x=q,y=h,kind=kind)
				ax.plot(q,f(q),
					label='usgs',c='g', linewidth=5)

		if hand: # Plot interpolated HAND rating curve
			f = self.interp(x=self.handq,y=self.handh,kind=kind)
			ax.plot(self.handq,f(self.handq),
				label='hand',c='b', linewidth=5)

		# Plot graph
		fig.set_size_inches(20,16, forward=True)
		plt.gca().set_xlim(left=0,right=self.max_disch)
		plt.gca().set_ylim(bottom=0,top=self.max_stage)
		ax.set_xticks(ax.get_xticks()[::2])
		ax.set_yticks(ax.get_yticks()[::2])
		title = 'COMID {0}'.format(self.comid)
		ax.set_title(title, y=1.04, fontsize=56)
		plt.xlabel('Q (cfs)',fontsize=56)
		plt.ylabel('H (ft)',fontsize=56)
		ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
		plt.rc('font', size=56)
		plt.legend(loc='upper left',fontsize=40)
		plt.tick_params(axis='both',labelsize=56)
		plt.grid()
		if filename:
			fig.savefig('results/rc_comid_{0}.png'.format(self.comid))
		else: 
			# mng = plt.get_current_fig_manager()
			# mng.resize(*mng.window.maxsize())
			plt.show()

if __name__ == '__main__':
	# Collect and pre-process datasets
	hand_curves = 'oniondata/handratingcurves.nc'
	hand_curves_idx = 'oniondata/handrc_idx.csv'
	
	hand_props = 'oniondata/OnionCreek.nc'
	hand_props_idx = 'oniondata/handnc_idx.csv'

	xsintersect = pandas.read_csv('oniondata/xsdata/xsintersect.csv',
		usecols=['COMID','ProfileM','RiverStation'])
	xsintersect = xsintersect.astype(str)
	xsrating = pandas.read_csv('oniondata/xsdata/xsrating.csv',
		usecols=['Stage_Height_ft_','Discharge_cfs_','RiverStation','XSID'])
	xsrating = xsrating.astype(str)
	xs = pandas.merge(xsrating,xsintersect,on='RiverStation')
	
	idlookup = pandas.read_csv('oniondata/streamgages.csv',usecols=['SOURCE_FEA','FLComID'])
	idlookup['SOURCE_FEA'] = idlookup['SOURCE_FEA'].apply(lambda x: '{0:0>8}'.format(x))
	
	comids = idlookup.FLComID.unique()

	# Instantiate RCDist class for each comid in watershed
	for comid in comids:
		# comid = 5781369
		try: 
			rcdist = RCDist(comid,hand_curves,hand_curves_idx,hand_props,hand_props_idx,xs,idlookup)
			print 'COMID {0} Plotted Successfully!'.format(str(comid))
		except TypeError: 
			print 'COMID {0} XS Error'.format(str(comid))
			continue
		except IndexError: 
			print 'COMID {0} RC Error'.format(str(comid))
			continue

		# Retrieve nstats
		# print rcdist.nstats(filename=False,upto=83)
		# print rcdist.handlen

		# Plot rating curves from data
		rcdist.plot_rc(filename=True,hand=False,usgs=False,xs=True,xsapprox=True,
			ci=False,raw=True,kind='linear',alpha=0.05,div=5)

# ************
# TODO
# ************
# Retrieve all possible data (particularly nstats) for 
# 	all stream reaches in the onion creek watershed.
# 	This will require computation of new _idx.csv files 
# 	and edits to the try/except failure conditions of
# 	the RCDist class, particularly regarding the XS try/except. 