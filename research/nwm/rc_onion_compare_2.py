# Paul J. Ruess
# University of Texas at Austin
# Fall, 2016

from pynwm import nwm
from netCDF4 import Dataset
import pandas
import scipy
import scipy.interpolate
from scipy.stats import norm
from collections import Counter

class RCData:

	def __init__(self, comid, ncdf, idx, xs, idlookup):
		"""Provides hand, xs, and usgs rating curve data for the
		specified comid
		'comid' - comid for which data is desired
		'ncdf' - NetCDF file containing HAND rating curve data
		'idx' - csv containing indices of HAND rating curves desired
		'xs' - csv containing xs data (must have profile, xsid, and rating curves)
		'idlookup' - csv lookup table between comid and usgsid"""
		self.comid = comid
		self.ncdf = Dataset(ncdf,'r')
		idx = pandas.read_csv(idx)
		self.idx = idx.loc[ idx['comid'] == self.comid ]['index'].values[0]
		self.get_handrc()
		self.idlookup = idlookup
		self.usgsid = self.idlookup.loc[self.idlookup['FLComID'] == self.comid]['SOURCE_FEA'].values[0]
		self.usgsrc = 'http://waterdata.usgs.gov/nwisweb/get_ratings?file_type=exsa&site_no={0}'
		if self.get_usgsrc() == 0: # Fetch self.usgsq and self.usgsh
			raise IndexError
		self.xs = xs[ xs['COMID'] == str(comid)]
		if self.get_xs() == None: 
			raise TypeError

	def get_handrc(self): 
		""" Initializes self.handq (cfs) and self.handh (ft) """
		handq = self.ncdf.variables['Q_cfs']
		handh = self.ncdf.variables['H_ft']
		handc = self.ncdf.variables['COMID']
		if handc[self.idx] == self.comid:
			self.handq = handq[self.idx]
		self.handh = handh

	def get_usgsrc(self):
		import urllib
		import re
		""" Initializes self.usgsq and self.usgsh """
		urlfile = urllib.urlopen(self.usgsrc.format(str(self.usgsid)))
		urllines = urlfile.readlines()
		findData = False
		usgsq = scipy.array([])
		usgsh = scipy.array([])
		for j in range(len(urllines)):
			line = urllines[j]
			if not findData and not re.search('[a-zA-Z]',line): # No letters
				findData = True
			if findData and float(line.split('\t')[2]) >= 1: # Remove data where Q < 1
				current = line.split('\t')
				usgsq = scipy.append( usgsq, float(current[2]) )
				# apply shift in [1] to stage height
				usgsh = scipy.append( usgsh, (float(current[0]) - float(current[1])) )
		shift = usgsh[0]
		self.usgsh = (usgsh - shift) # Normalize usgsh over bottom depth
		self.usgsq = usgsq

	def get_xs(self):
		""" Retrieve xs information to populate self.xscurves. 
		Also create and populate self.max_q_query with maximum q 
		value for querying interpolated rating curves."""
		xscurves = []
		self.max_q_query = 0
		self.max_disch = 0
		self.max_stage = 0

		# Retrieve xs information and populate self.xscurves
		stations = self.xs['RiverStation'].unique()
		for i, rs in enumerate( stations ):
			a = self.xs[self.xs['RiverStation'] == rs]['Stage_Height_ft_'].values
			s = [item for item, count in Counter(a).iteritems() if count > 1]
			xsids = scipy.array([])
			if s: xsids = self.xs[self.xs['RiverStation'] == rs]['XSID'].unique()
			if len(xsids) > 0:
				continue
				# for xsid in xsids:
				# 	current = self.xs[ self.xs['XSID'] == xsid ]
				# 	prof = current['ProfileM'].unique()[0] # location of xs relative to river reach
				# 	disch = map(float,current['Discharge_cfs_'].values) # xs disch vals
				# 	stage = map(float,current['Stage_Height_ft_'].values) # xs stage vals
			else: 
				current = self.xs[ self.xs['RiverStation'] == rs ]
				prof = current['ProfileM'].unique()[0] # location of xs relative to river reach
				disch = map(float,current['Discharge_cfs_'].values) # xs disch vals
				stage = map(float,current['Stage_Height_ft_'].values) # xs stage vals

			# Find max q value for querying interpolations
			# Find max disch value for plotting x_axis
			max_disch = int( scipy.amax(disch) ) 
			if i == 0:
				self.max_q_query = max_disch
				self.max_disch = max_disch
			elif max_disch < self.max_q_query: 
				self.max_q_query = max_disch
			elif max_disch > self.max_disch: 
				self.max_disch = max_disch

			# Find max stage value for plotting y_axis
			max_stage = int( scipy.amax(stage) )
			if i == 0: 
				self.max_stage = max_stage
			elif max_stage > self.max_stage:
				self.max_stage = max_stage

			pack = (prof,zip(disch, stage)) # pack xs profile name w/ disch & stage vals
			xscurves.append( pack )
		if len(xscurves) != 0:
			self.xscurves = xscurves
			return 1

class RCDist(RCData):

	def get_stage(self,q):
		"""Given discharge q, query interpolated rating curves
		and return associated stage-heights."""
		other_stages = []
		for i in self.xscurves:
			disch, stage = zip(*i[1])
			print q
			print disch,stage
			interp_other = scipy.interpolate.interp1d(disch,stage,kind='linear') # kind='cubic'
			other_stages.append( interp_other(q) )		
		self.interp_hand = scipy.interpolate.interp1d(self.handq,self.handh,kind='linear') # kind='cubic'
		hand_stage = self.interp_hand(q)
		return (hand_stage, other_stages)

	def get_ci(self, alpha=0.05, div=5):
		"""Get upper- and lower-bounds (confidence interval)
		of normal distribution from xs stage_heights at 'div'
		evenly spaced intervals. 
		'alpha' - statistical alpha value
		'div' - number of intervals"""
		print self.max_q_query
		step = self.max_q_query/div
		ci_qvals = scipy.arange( 0 , (self.max_q_query+step), step )

		ubounds = [] # upper-bounds
		lbounds = [] # lower-bounds
		for q in ci_qvals: 
			hand_stage = self.get_stage(q)[0] # hand_stage
			other_stages = self.get_stage(q)[1] # other_stages
			mean = scipy.average(other_stages)
			stdev = scipy.std(other_stages)
			z = norm.ppf( (1-alpha/2), scale=stdev ) # z, mean = 0, 2-sided
			lb = mean-z # lower bound
			ub = mean+z # upper bound
			ubounds.append(ub)
			lbounds.append(lb)
		self.ci_qvals = ci_qvals
		self.ubounds = scipy.nan_to_num( ubounds )
		self.lbounds = scipy.nan_to_num( lbounds )

	def plot_rc(self,hand=True,xs=True,ci=True,alpha=0.05,div=5,kind='linear'):
		"""Plot HAND and xs rating curves with confidence intervals
		'hand' - plot hand rating curve [T/F]
		'xs' - plot xs rating curves [T/F]
		'ci' - plot confidence intervals [T/F]
		'alpha' - alpha for confidence intervals [float(0.0,1.0)]
		'div' - number of intervals for confidence interval [R]"""
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots() # get figure and axes for plotting

		# Plot all linearly-interpolated XS rating curves
		for i in self.xscurves:
			disch, stage = zip(*i[1])
			f = scipy.interpolate.interp1d(disch,stage,kind=kind)
			ax.plot(disch,f(disch),c='grey', linewidth=1)#, label= i[0])

		# Plot confidence interval bounds
		self.get_ci(alpha,div) # get upper- and lower-bound variables
		# upper bounds
		f = scipy.interpolate.interp1d(self.ci_qvals,self.ubounds,kind=kind)
		ax.plot(self.ci_qvals,f(self.ci_qvals),
			label='{0}%CI'.format( int((1-alpha)*100) ),c='orange', linewidth=5)#,s=400,marker='^'
		# lower bounds
		f = scipy.interpolate.interp1d(self.ci_qvals,self.lbounds,kind=kind)
		ax.plot(self.ci_qvals,f(self.ci_qvals),
			label='{0}%CI'.format( int((1-alpha)*100) ),c='orange', linewidth=5)#,s=400,marker='^'

		# Plot linearly-interpolated HAND rating curve
		f = scipy.interpolate.interp1d(self.usgsq,self.usgsh,kind=kind)
		ax.plot(self.usgsq,f(self.usgsq),
			label='usgs',c='g', linewidth=5)#,s=400,marker='^'

		# Plot linearly-interpolated HAND rating curve
		f = scipy.interpolate.interp1d(self.handq,self.handh,kind=kind)
		ax.plot(self.handq,f(self.handq),
			label='hand',c='b', linewidth=5)#,s=400,marker='^'

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
		# fig.savefig('rc_comid_{0}.png'.format(self.comid))
		plt.show()

class MapFlood(RCData):

	def nwm_stage(self,forecast):
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

if __name__ == '__main__':
	ncdf = 'handratingcurves.nc'
	idx = 'handrc_idx.csv'

	xsintersect = pandas.read_csv('/home/paul/Desktop/xsintersect.csv',
		usecols=['COMID','ProfileM','RiverStation'])
	xsintersect = xsintersect.astype(str)
	xsrating = pandas.read_csv('/home/paul/Desktop/xsrating.csv',
		usecols=['Stage_Height_ft_','Discharge_cfs_','RiverStation','XSID'])
	xsrating = xsrating.astype(str)
	xs = pandas.merge(xsrating,xsintersect,on='RiverStation')
	idlookup = pandas.read_csv('streamgages.csv',usecols=['SOURCE_FEA','FLComID'])
	idlookup['SOURCE_FEA'] = idlookup['SOURCE_FEA'].apply(lambda x: '{0:0>8}'.format(x))

	for i in range(len(idlookup)):
		# i = 4
		comid = idlookup['FLComID'][i]
		try: 
			rcdist = RCDist(comid,ncdf,idx,xs, idlookup)
		except TypeError: 
			usgsid = idlookup['SOURCE_FEA'][i]
			print 'No cross-sections for COMID {0} / USGSID {1}'.format(str(comid),str(usgsid))
			continue
		except IndexError: 
			usgsid = idlookup['SOURCE_FEA'][i]
			print 'No USGS rating curve for COMID {0} / USGSID {1}'.format(str(comid),str(usgsid))
			continue

		rcdist.plot_rc(hand=True,xs=True,ci=True,alpha=0.05,div=5,kind='linear')
		
		# handrc = RCData(comid,ncdf,idx)
		# # print handrc.nwm_stage('short_range')
		# handrc.plot_rc()