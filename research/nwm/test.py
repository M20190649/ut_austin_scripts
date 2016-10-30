from pynwm import nwm
from netCDF4 import Dataset
import pandas
import scipy
import scipy.interpolate

class MapFlood:

	def __init__(self, comid, rc, hand):
		self.comid = comid



class HandRC:

	def __init__(self, comid, ncdf, idx):
		self.comid = comid
		self.ncdf = Dataset(ncdf,'r')
		idx = pandas.read_csv(idx)
		self.idx = idx.loc[ idx['comid'] == self.comid ]['index'].values[0]
		self.get_handrc()

	def get_handrc(self): 
		""" Initializes self.handq (cfs) and self.handh (ft) """
		handq = self.ncdf.variables['Q_cfs']
		handh = self.ncdf.variables['H_ft']
		handc = self.ncdf.variables['COMID']
		if handc[self.idx] == self.comid:
			self.handq = handq[self.idx]
		self.handh = handh

	def get_stage(self,forecast):
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

	def plot_rc(self):
		import matplotlib.pyplot as plt
		import matplotlib.ticker as ticker

		fig, ax = plt.subplots()

		f = scipy.interpolate.interp1d(self.handq,self.handh) # kind=cubic

		fig.set_size_inches(20,16, forward=True)

		ax.plot( self.handq, f(self.handq), label='hand' ) #,c='none',s=400,marker='^'

		plt.gca().set_xlim(left=0)
		plt.gca().set_ylim(bottom=0)
		ax.set_xticks(ax.get_xticks()[::2])
		ax.set_yticks(ax.get_yticks()[::2])
		title = 'COMID {0}'.format(self.comid)
		ax.set_title(title, y=1.04, fontsize=56)
		# # plt.title(title, y=1.04, fontsize=64)
		plt.xlabel('Q (cfs)',fontsize=56)
		plt.ylabel('H (ft)',fontsize=56)
		ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
		plt.rc('font', size=56)
		plt.legend(loc='lower right',fontsize=56)
		plt.tick_params(axis='both',labelsize=56)
		# plt.grid()
		# fig.savefig('rc_comid_{0}.png'.format(self.comid))
		plt.show()

if __name__ == '__main__':
	idlookup = pandas.read_csv('streamgages.csv',usecols=['SOURCE_FEA','FLComID'])
	idlookup['SOURCE_FEA'] = idlookup['SOURCE_FEA'].apply(lambda x: '{0:0>8}'.format(x))
	ncdf = 'handratingcurves.nc'
	idx = 'handrc_idx.csv'

	for i in range(len(idlookup)):
		comid = idlookup['FLComID'][i]
		handrc = HandRC(comid,ncdf,idx)
		print handrc.get_stage('short_range')
		handrc.plot_rc()





# 297, 984

# print handrc.get_stage(974)



# for s in series:
# 	dates = s['dates']
# 	for i, v in enumerate(s['values']):
# 		print dates[i].strftime('%y-%m-%d %H'), '\t', v