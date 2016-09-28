from __future__ import division
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas
import urllib
import re
from math import sqrt
import scipy

class CompareRC:

	def __init__(self,comid,idlookup,handrc,handrcidx,handnetcdf,handnetcdfidx):
		self.comid = comid
		self.idlookup = idlookup
		self.handnetcdf = handnetcdf
		self.handrc = handrc
		self.usgsid = self.idlookup.loc[self.idlookup['FLComID'] == self.comid]['SOURCE_FEA'].values[0]
		self.usgsrc = 'http://waterdata.usgs.gov/nwisweb/get_ratings?file_type=exsa&site_no={0}'
		self.handnetcdfidx = handnetcdfidx.loc[handnetcdfidx['comid'] == self.comid]['index'].values[0]
		self.handrcidx = handrcidx.loc[handrcidx['comid'] == self.comid]['index'].values[0]

	def get_values(self):
		if self.get_usgsrc() == 0:
			return
		self.get_usgsrc() # Fetch self.usgsq and self.usgsh
		self.get_handrc() # Fetch self.handq and self.handh
		self.get_handnetcdf() # Fetch self.handarea, self.handrad, self.handslope, and self.handstage

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
		try:
			self.bottomdepth = usgsh[0] # Set first Q = 1.0 equal to bottom depth
		except IndexError:
			print 'USGS Rating Curve does not exist for USGSID {0}'.format(str(self.usgsid))
			return 0
		self.usgsh = (usgsh - self.bottomdepth) # Normalize usgsh over bottom depth
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

	def data_to_csv(self):
		import itertools
		import csv
		if self.get_usgsrc() == 0:
			return
		self.get_values() # Fetch usgsq,usgsh,handq,handh,handarea,handrad,handslope, handstage
		self.get_usgs_n() # Fetch self.usgsroughness
		self.calc_handrc(self.usgsroughness) # Fetch self.handdisch and self.handstage
		info = [self.comid,self.usgsid,self.handslope]
		rows = itertools.izip_longest(info,self.handstage,self.handarea, \
			self.handrad,self.usgsh,self.usgsq,fillvalue='')
		with open('onionck/results/newcsvdata/{0}_data.csv'.format(self.comid), 'w') as f:
			csv.writer(f).writerows(rows)
		f.close()

	def get_usgs_n(self):
		if self.get_usgsrc() == 0:
			return
		self.get_values() # Fetch usgsq,usgsh,handq,handh,handarea,handrad,handslope, handstage
		
		# Find indices for integer stageheight values in usgsh, and apply to usgsq
		usgsidx = scipy.where(scipy.equal(scipy.mod(self.usgsh,1),0)) # Find indices of integer values in usgsh
		usgsh = self.usgsh[usgsidx]
		usgsq = self.usgsq[usgsidx]

		# Find indices where usgsh[usgsidx] occur in handstage, and apply to handarea and handrad
		handidx = scipy.where(scipy.in1d(self.handstage,usgsh))
		area = self.handarea[handidx]
		hydrad = self.handrad[handidx]

		# Remove usgsq values for duplicate usgsh heights (keep first instance only)
		if usgsh.shape != area.shape:
			for i in range(usgsh.shape[0]):
				if i == 0: pass
				elif usgsh[i] == usgsh[i-1]:
					usgsq = scipy.delete(usgsq,i)

		# Calculate average manning's n after converting discharge units
		disch = usgsq #*0.0283168 # Convert cfs to cms
		self.usgsroughness_array = self.mannings_n(area=area,hydrad=hydrad,slope=self.handslope,disch=disch)
		self.usgsroughness = scipy.average(self.usgsroughness_array)
		print 'Average roughness: {0:.2f}'.format(self.usgsroughness)

	def plot_rcs(self):
		if self.get_usgsrc() == 0:
			return
		self.get_values() # Fetch usgsq,usgsh,handq,handh,handarea,handrad,handslope, handstage
		self.get_usgs_n() # Fetch self.usgsroughness
		self.calc_handrc(self.usgsroughness) # Fetch self.handdisch
		fig, ax = plt.subplots()
		fig.set_size_inches(20,16, forward=True)
		ax.scatter(self.usgsq,self.usgsh,c='b',s=100,label='usgs')
		ax.scatter(self.handq,self.handh,c='r',s=100,label='hand')
		ax.scatter(self.handdisch,self.handstage,c='g',s=100,label='hand_calc')
		plt.gca().set_xlim(left=0)
		plt.gca().set_ylim(bottom=0)
		ax.set_title('USGS {0}, COMID {1}'.format(str(self.usgsid),str(self.comid)),fontsize=32)
		plt.xlabel('Q (cfs)',fontsize=28)
		plt.ylabel('H (ft)',fontsize=28)
		ax.text((self.handq[-1]*0.8),55,"Manning's n: {0:.2f}".format(self.usgsroughness),horizontalalignment='left',
			fontsize=24,bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
		plt.legend(loc='upper left',fontsize=24)
		plt.grid()
		plt.show()
		fig.savefig('onionck/results/autoresults/rc_comid_{0}.pdf'.format(self.comid))

if __name__ == '__main__':

	idlookup = pandas.read_csv('streamgages.csv',usecols=['SOURCE_FEA','FLComID'])
	idlookup['SOURCE_FEA'] = idlookup['SOURCE_FEA'].apply(lambda x: '{0:0>8}'.format(x))
	handrc = Dataset('handratingcurves.nc', 'r')
	handrcidx = pandas.read_csv('handrc_idx.csv')
	handnetcdfidx = pandas.read_csv('handnc_idx.csv')
	handnetcdf = Dataset('onionck/OnionCreek.nc', 'r')

	for i in range(len(idlookup)):
		comid = idlookup['FLComID'][i]
		rcs = CompareRC(comid,idlookup,handrc,handrcidx,handnetcdf,handnetcdfidx)
		rcs.plot_rcs()
		# rcs.data_to_csv()