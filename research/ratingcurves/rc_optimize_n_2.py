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
import csv
import itertools

class RCData:

	def __init__(self, comid, hand_curves, hand_curves_idx, 
		hand_props, hand_props_idx, idlookup, sprnt_data):
		"""Provides hand, xs, and usgs rating curve data for the specified comid.
		'comid' - comid for which data is desired
		'hand_curves' - NetCDF file containing HAND rating curve data
		'hand_curves_idx' - csv containing indices of HAND rating curves desired
		'hand_props' - NetCDF file containing HAND hydraulic property data
		'hand_props_idx' - csv containing indices of HAND hydraulic properties desired
		'xs' - csv containing xs data (must have profile, xsid, and rating curves)
		'idlookup' - csv lookup table between comid and usgsid"""
		self.comid = comid
		print "Retrieving data for comid {0}".format(self.comid)

		self.hand_curves = Dataset(hand_curves, 'r')

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

		self.usgsids = self.idlookup.loc[self.idlookup['FLComID'] == self.comid]['SOURCE_FEA'].values
		self.usgsrc = 'http://waterdata.usgs.gov/nwisweb/get_ratings?file_type=exsa&site_no={0}'

		self.get_usgsrc() # Fetch usgs stage and disch values

		self.sprnt_df = pandas.read_csv(sprnt_data.format(comid))
		self.get_sprnt()

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
			self.handarea = handarea[self.hand_props_idx]*(3.28084**2) # Convert sqm to sqft
			self.handrad = handrad[self.hand_props_idx]*3.28084 # Convert m to ft
			self.handslope = handslope[self.hand_props_idx] # unitless
			self.handlen = handlen[self.hand_props_idx]*3.28084 # Convert m to ft
		handstage = scipy.array(handstage)*3.28084 # Convert m to ft
		self.handstage = scipy.rint(handstage) # Round to nearest int

	def get_hand_curves(self): 
		""" Initializes self.handq [cfs] and self.handh [ft]."""
		handq = self.hand_curves.variables['Q_cfs']
		handh = self.hand_curves.variables['H_ft']
		handc = self.hand_curves.variables['COMID']
		if handc[self.hand_curves_idx] == self.comid:
			self.handq = handq[self.hand_curves_idx]
		self.handh = handh

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

	def get_sprnt(self):
		self.sprnth = self.sprnt_df[' Depth(m)'].values # Depth(m)
		self.sprnth = self.sprnth*3.28084 # Convert m to ft
		self.sprntq = self.sprnt_df['Flowrate(m^3)'].values # Flowrate(m^3)
		self.sprntq = self.sprntq*(3.28084**3) # Convert m^3 to ft^3

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

	def mannings_n(self,area,hydrad,slope,disch):
		""" Calculates manning's roughness from discharge. 
		'area' - self.handarea (wet area),
		'hydrad' - self.handrad (hydraulic radius),
		'slope' - self.handslope (bed slope), and
		'disch' - any discharge values"""
		res = 1.49*area*scipy.power(hydrad,(2/3.0))*scipy.sqrt(slope)/disch.T
		return res.T

	def optimize_n(self):
		usgs_hlist = []
		usgs_hset = set()
		usgs_qlist = []
		for h in self.handstage: 
			res = min(enumerate(self.usgsh[0]), key=lambda x:abs(x[1]-h))
			if res[1] not in usgs_hset:
				usgs_hlist.append(res[1])
				usgs_hset.add(res[1])
				usgs_qlist.append(self.usgsq[0][res[0]])
		usgs_hlist = scipy.array(usgs_hlist[:-1])
		usgs_qlist = scipy.array(usgs_qlist[:-1])
		area = self.handarea[:len(usgs_qlist)]
		hydrad = self.handrad[:len(usgs_qlist)]
		opt_n = self.mannings_n(area=area,hydrad=hydrad,slope=self.handslope,disch=usgs_qlist)
		return zip(usgs_hlist,usgs_qlist,)

	def plot_rc(self,save=False,hand=True,usgs=True,sprnt=True,
		dist=5000,kind='power',alpha=0.05,div=5):
		"""Plot HAND and xs rating curves with confidence intervals
		'hand' - plot hand rating curve [T/F]
		'xs' - plot xs rating curves [T/F]
		'xsapprox' - plot xs rating curve approximation from n-value averages [T/F]
		'ci' - plot confidence intervals [T/F]
		'alpha' - alpha for confidence intervals [float(0.0,1.0)]
		'div' - number of intervals for confidence interval [R]"""

		fig, ax = plt.subplots() # get figure and axes for plotting

		if usgs: # Plot interpolated USGS rating curve
			# Plot curves
			for q,h in zip(self.usgsq,self.usgsh):
				if kind == 'cubic': 
					print 'USGS interpolation plotted as power-law fit'
					f = self.interp(x=q,y=h,kind='power')
				else: 
					f = self.interp(x=q,y=h,kind=kind)
				ax.plot(q,f(q),
					label='usgs',c='g', linewidth=5)

		if hand: # Plot interpolated HAND rating curve
			# Plot curves
			f = self.interp(x=self.handq,y=self.handh,kind=kind)
			ax.plot(self.handq,f(self.handq),
				label='hand',c='b', linewidth=5)

		if sprnt:
			f = self.interp(x=self.sprntq,y=self.sprnth,kind=kind)
			ax.plot(self.sprntq,f(self.sprntq),
				label='sprnt',c='y', linewidth=5)					

		# Plot graph
		fig.set_size_inches(20,16, forward=True)
		plt.gca().set_xlim(left=0,right=self.handq[-1])
		plt.gca().set_ylim(bottom=0,top=self.handh[-1])
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

		if save:
			fig.savefig(fname)
			plt.clf()

		if not save: 
			pass
			mng = plt.get_current_fig_manager()
			mng.resize(*mng.window.maxsize())
			plt.show()
			plt.clf()

if __name__ == '__main__':
	
	# Path to HAND files
	hand_curves = 'oniondata/handratingcurves.nc'
	hand_curves_idx = 'oniondata/handrc_idx.csv'
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

	# Path to SPRNT files
	sprnt_data = 'oniondata/sprnt_rcs/{0}_SPRNT_RatingCurve.csv'
	
	# Get lists of COMIDs to iterate over
	comids = idlookup.FLComID.unique()

	# Override with reasonable-looking comids (from multi-boxplots)
	comids = [5781431,5781477,5781369]

	# Instantiate RCDist class for each comid in watershed
	for comid in comids:
		try: 
			rcdist = RCDist(comid,hand_curves, hand_curves_idx, 
				hand_props,hand_props_idx,idlookup,sprnt_data)
			print 'COMID {0} Collected Successfully!'.format(str(comid))
		except TypeError: 
			print 'COMID {0} XS Error'.format(str(comid))
			continue
		except IndexError: 
			print 'COMID {0} RC Error'.format(str(comid))
			continue
		except KeyError:
			print 'COMID {0} SPRNT Error'.format(str(comid))
			continue

		print rcdist.optimize_n()

		# Plot rating curves from data
		rcdist.plot_rc(save=False,hand=True,usgs=True,sprnt=False,
			dist=100000000,kind='linear',alpha=0.05,div=5)