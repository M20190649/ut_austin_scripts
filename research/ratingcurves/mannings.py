from __future__ import division
from netCDF4 import Dataset
import pandas
import urllib
import re
from math import sqrt
import scipy

class CompareRC:

	def __init__(self,comid,idlookup,handrc,handrcidx,handnetcdf,handnetcdfidx, en=False):
		self.comid = comid
		self.idlookup = idlookup
		self.usgsid = self.idlookup.loc[self.idlookup['FLComID'] == self.comid]['SOURCE_FEA'].values[0]
		self.usgsrc = 'http://waterdata.usgs.gov/nwisweb/get_ratings?file_type=exsa&site_no={0}'
		if self.get_usgsrc() == 0: # Fetch self.usgsq and self.usgsh
			raise IndexError
		self.handnetcdf = handnetcdf
		self.handrc = handrc
		self.handnetcdfidx = handnetcdfidx.loc[handnetcdfidx['comid'] == self.comid]['index'].values[0]
		self.handrcidx = handrcidx.loc[handrcidx['comid'] == self.comid]['index'].values[0]
		self.get_handrc() # Fetch self.handq and self.handh
		self.get_handnetcdf() # Fetch self.handarea, self.handrad, self.handslope, and self.handstage
		self.get_in_common()

		if en:
			self.usgsroughness = en
		else:
			self.get_usgs_n() # Fetch self.usgsroughness
		print 'Average roughness: {0:.2f}'.format(self.usgsroughness)
		self.calc_handrc(self.usgsroughness) # Fetch self.handdisch and self.handstage
		self.handstageint = self.handstage[self.handidx]
		self.handdischint = self.handdisch[self.handidx]
		self.min_leastsq()

	def get_in_common(self):		
		# Find indices for integer stageheight values in usgsh, and apply to usgsq
		usgsidx = scipy.where(scipy.equal(scipy.mod(self.usgsh,1),0)) # Find indices of integer values in usgsh
		usgshint = self.usgsh[usgsidx] # Integers in usgsh
		usgsqint = self.usgsq[usgsidx] # Integers in usgsq

		# Find indices where usgshint[usgsidx] occur in handstage, and apply to handarea and handrad
		handidx = scipy.where(scipy.in1d(self.handstage,usgshint))
		handareaint = self.handarea[handidx]
		handradint = self.handrad[handidx]

		# Remove usgsqint values for duplicate usgshint heights (keep first instance only)
		if usgshint.shape != handareaint.shape:
			for i in range(usgshint.shape[0]):
				if i == 0: pass
				elif usgshint[i] == usgshint[i-1]:
					usgsqint = scipy.delete(usgsqint,i)

		self.usgshint = usgshint
		self.usgsqint = usgsqint
		self.handareaint = handareaint
		self.handradint = handradint
		self.usgsidx = usgsidx
		self.handidx = handidx

	def get_usgsrc(self):
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

	def get_handrc(self): 
		""" Initializes self.handq (cfs) and self.handh (ft) """
		handq = self.handrc.variables['Q_cfs']
		handh = self.handrc.variables['H_ft']
		handc = self.handrc.variables['COMID']
		if handc[self.handrcidx] == self.comid:
			self.handq = handq[self.handrcidx]
		self.handh = handh

	def get_handnetcdf(self):
		"""Initializes self.handarea (sqmeters), self.handrad (m), 
			self.handslope (-), and self.handstage (ft) """
		handc = self.handnetcdf.variables['COMID']
		handslope = self.handnetcdf.variables['Slope']
		handarea = self.handnetcdf.variables['WetArea']
		handrad = self.handnetcdf.variables['HydraulicRadius']
		handstage = self.handnetcdf.variables['StageHeight']
		if handc[self.handnetcdfidx] == self.comid:
			self.handarea = handarea[self.handnetcdfidx]*10.7639 # Convert sqm to sqft
			self.handrad = handrad[self.handnetcdfidx]*3.28084 # Convert m to ft
			self.handslope = handslope[self.handnetcdfidx]
		handstagenew = scipy.array([])
		for i in handstage:
			handstagenew = scipy.append(handstagenew, handstage)
		self.handstage = handstagenew
		self.handstage = self.handstage[:49]*3.28084 # Convert m to ft
		self.handstage = scipy.rint(self.handstage) # Round to nearest int, to clean up conversion

	def mannings_q(self,area,hydrad,slope,en):
		""" Calculates discharge from manning's roughness using Wet Area = self.handarea, 
			Hydraulic Radius = self.handrad, and Slope = self.handslope """
		return 1.49*area*scipy.power(hydrad,(2/3.0))*sqrt(slope)/en

	def mannings_n(self,area,hydrad,slope,disch):
		""" Calculates manning's roughness from discharge using Wet Area = self.handarea, 
			Hydraulic Radius = self.handrad, and Slope = self.handslope """
		return 1.49*area*scipy.power(hydrad,(2/3.0))*sqrt(slope)/disch

	def calc_handrc(self,en):
		""" Initializes self.handdisch (cfs)"""
		handdisch = self.mannings_q(area=self.handarea,hydrad=self.handrad,slope=self.handslope,en=en)
		self.handdisch = handdisch[:49]

	def convert_to_english(self):
		pass

	def calc_leastsq(self):
		dif = self.usgsqint - self.handdischint
		sqdif = dif**2
		leastsq = scipy.average(sqdif)
		print 'Least-Squares:', leastsq
		return leastsq

	def get_usgs_n(self):
		self.get_in_common()

		# Calculate average manning's n after converting discharge units
		area = self.handareaint
		hydrad = self.handradint
		slope = self.handslope
		disch = self.usgsqint #*0.0283168 # Convert cfs to cms
		self.usgsroughness_array = self.mannings_n(area=area,hydrad=hydrad,slope=slope,disch=disch)
		self.usgsroughness = scipy.average(self.usgsroughness_array)

	def min_leastsq(self):
		self.calc_leastsq()

	def data_to_csv(self):
		import itertools
		import csv
		info = [self.comid,self.usgsid,self.handslope]
		rows = itertools.izip_longest(info,self.handstage,self.handarea, \
			self.handrad,self.usgsh,self.usgsq,fillvalue='')
		print list(rows)
		# with open('onionck/results/newcsvdata/{0}_data.csv'.format(self.comid), 'w') as f:
		# 	csv.writer(f).writerows(rows)
		# f.close()

	def plot_rcs(self):
		import matplotlib.pyplot as plt
		import matplotlib.ticker as ticker
		fig, ax = plt.subplots()
		fig.set_size_inches(20,16, forward=True)
		ax.scatter(self.handq,self.handh,c='none',s=400,marker='^',label='hand')
		ax.scatter(self.handdisch,self.handstage,c='none',s=400,marker='s',label='hand_fit')
		ax.scatter(self.usgsq,self.usgsh,c='black',s=400,marker='o',label='usgs')
		plt.gca().set_xlim(left=0)
		plt.gca().set_ylim(bottom=0)
		ax.set_xticks(ax.get_xticks()[::2])
		ax.set_yticks(ax.get_yticks()[::2])
		# ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
		# tick_spacing = 400000
		# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
		ax.set_xticks([400000,800000,1200000])
		ax.set_yticks([0,20,40,60])

		# title = 'USGS {0}, COMID {1}'.format(str(self.usgsid),str(self.comid))
		# ax.set_title(fontsize=56)
		# plt.title(title, y=1.04, fontsize=64)
		plt.xlabel('Q (cfs)',fontsize=56)
		plt.ylabel('H (ft)',fontsize=56)
		# ax.text((self.handq[-1]*0.8),55,"Manning's n: {0:.2f}".format(self.usgsroughness),horizontalalignment='left',
		# 	fontsize=24,bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
		# xformat = ticker.FuncFormatter( lambda x, p: format(int(x), ',') )
		# ax.xaxis.set_major_formatter(xformat)
		ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
		plt.rc('font', size=56)

		plt.legend(loc='lower right',fontsize=56)
		plt.tick_params(axis='both',labelsize=56)
		# plt.grid()
		# fig.savefig('rc_comid_{0}.png'.format(self.comid))
		plt.show()

if __name__ == '__main__':

	idlookup = pandas.read_csv('oniondata/streamgages.csv',usecols=['SOURCE_FEA','FLComID'])
	idlookup['SOURCE_FEA'] = idlookup['SOURCE_FEA'].apply(lambda x: '{0:0>8}'.format(x))
	handrc = Dataset('oniondata/handratingcurves.nc', 'r')
	handrcidx = pandas.read_csv('oniondata/handrc_idx.csv')
	handnetcdfidx = pandas.read_csv('oniondata/handnc_idx.csv')
	handnetcdf = Dataset('oniondata/OnionCreek.nc', 'r')
	en = None

	for i in range(len(idlookup)):
		i = 4
		comid = idlookup['FLComID'][i]
		try: 
			rcs = CompareRC(comid,idlookup,handrc,handrcidx,handnetcdf,handnetcdfidx, en=0.35)
		except IndexError: 
			usgsid = idlookup['SOURCE_FEA'][i]
			print 'USGS Rating Curve does not exist for USGSID {0}'.format(str(usgsid))
			continue
		# rcs.calc_leastsq()
		# rcs.plot_rcs()

		rcs.data_to_csv()