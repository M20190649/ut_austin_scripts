from __future__ import division
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas
import urllib
import re
from math import sqrt
import scipy

class CompareRC:

	def __init__(self,idlookup,handrc,handrcidx,handnc,handncidx):
		self.idlookup = idlookup
		self.handnc = handnc
		self.handrc = handrc
		self.handrcidx = handrcidx
		self.handncidx = handncidx
		self.usgsrc = 'http://waterdata.usgs.gov/nwisweb/get_ratings?file_type=exsa&site_no={0}'

	def get_usgsrc(self,usgsid):
		urlfile = urllib.urlopen(self.usgsrc.format(str(usgsid)))
		urllines = urlfile.readlines()
		check = False
		usgsq = []
		usgsh = []
		for j in range(len(urllines)):
			line = urllines[j]
			if not re.search('[a-zA-Z]',line): # No letters
				check = True
			if check:
				usgsq.append( float(line.split('\t')[2]) )
				# apply shift in [1] to stage height
				usgsh.append( float(line.split('\t')[0]) - float(line.split('\t')[1]) )
		return (usgsq,usgsh)

	def get_handrc(self,comid): 
		handq = self.handrc.variables['Q_cfs']
		handh = self.handrc.variables['H_ft']
		handc = self.handrc.variables['COMID']
		idx = self.handrcidx.loc[self.handrcidx['comid'] == comid]['index'].values[0]
		if handc[idx] == comid:
			handq = handq[idx]
		return (handq,handh)

	def mannings(self,wetarea,hydraulicradius,slope,en):
		return wetarea*scipy.power(hydraulicradius,(2/3.0))*sqrt(slope)/en

	def get_handnc(self,comid,en=0.05):
		idx = self.handncidx.loc[self.handncidx['comid'] == comid]['index'].values[0]
		handstage = self.handnc.variables['StageHeight']
		handc = self.handnc.variables['COMID']
		handslope = self.handnc.variables['Slope']
		handarea = self.handnc.variables['WetArea']
		handrad = self.handnc.variables['HydraulicRadius']
		if handc[idx] == comid:
			handarea = handarea[idx]
			handrad = handrad[idx]
			handslope = handslope[idx]
		handdisch = self.mannings(wetarea=handarea,hydraulicradius=handrad,slope=handslope,en=en)
		handstage = handstage[:49]*3.28084 # Convert m to ft
		handdisch = handdisch[:49]*35.3147 # Convert cms to cfs
		return (handstage,handdisch)

	def get_usgs_n(self,usgsid):
		comid = self.idlookup.loc[self.idlookup['SOURCE_FEA'] == usgsid]['FLComID'].values[0]
		stage = self.handrc.variables['H_ft']
		usgsq,usgsh = get_usgsrc(usgsid)

	def plot_rcs(self,comid):
		usgsid = self.idlookup.loc[self.idlookup['FLComID'] == comid]['SOURCE_FEA'].values[0]
		usgsq,usgsh = self.get_usgsrc(usgsid)
		if usgsq == usgsh == []:
			print 'USGS Rating Curve does not exist for COMID {0}'.format(str(comid))
			return
		handq,handh = self.get_handrc(comid)
		en = 0.15
		stage,disch = self.get_handnc(comid,en)
		fig, ax = plt.subplots()
		fig.set_size_inches(20,16, forward=True)
		ax.scatter(usgsq,usgsh,c='b',s=100,label='usgs')
		ax.scatter(handq,handh,c='r',s=100,label='hand')
		ax.scatter(disch,stage,c='g',s=100,label='hand_calc')
		plt.gca().set_xlim(left=0)
		plt.gca().set_ylim(bottom=0)
		ax.set_title('USGS {0}, COMID {1}'.format(str(usgsid),str(comid)),fontsize=32)
		plt.xlabel('Q (cfs)',fontsize=28)
		plt.ylabel('H (ft)',fontsize=28)
		ax.text((handq[-1]*0.8),55,"Manning's n: {0}".format(str(en)),horizontalalignment='left',
			fontsize=24,bbox={'facecolor':'green', 'alpha':0.5, 'pad':10})
		plt.legend(loc='upper left',fontsize=24)
		plt.grid()
		plt.show()
		fig.savefig('manualresults/rc_comid_{0}.pdf'.format(comid))

if __name__ == '__main__':

	idlookup = pandas.read_csv('streamgages.csv',usecols=['SOURCE_FEA','FLComID'])
	idlookup['SOURCE_FEA'] = idlookup['SOURCE_FEA'].apply(lambda x: '{0:0>8}'.format(x))
	handrc = Dataset('handratingcurves.nc', 'r')
	handrcidx = pandas.read_csv('handrc_idx.csv')
	handncidx = pandas.read_csv('handnc_idx.csv')
	handnc = Dataset('onionck/OnionCreek.nc', 'r')

	rcs = CompareRC(idlookup,handrc,handrcidx,handnc,handncidx)

	for i in range(len(idlookup)):
		# rcs.get_usgsrc(idlookup['SOURCE_FEA'][i])
		# rcs.get_handrc(idlookup['FLComID'][i])
	# rcs.get_n(idlookup['FLComID'][0])
	# rcs.get_nhd_info(idlookup['FLComID'][0])
		rcs.plot_rcs(idlookup['FLComID'][i])
		# rcs.get_usgs_n(idlookup['SOURCE_FEA'][i])
	# rcs.get_handnc(idlookup['FLComID'][0])