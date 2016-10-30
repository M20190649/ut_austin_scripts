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

	def plot_rc(self):
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots()
		fig.set_size_inches(20,16, forward=True)

		f = scipy.interpolate.interp1d(self.handq,self.handh) # kind=cubic
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

class RCDist(HandRC):

	def __init__(self, comid, ncdf, idx, xs):
		self.comid = comid
		self.ncdf = Dataset(ncdf,'r')
		idx = pandas.read_csv(idx)
		self.idx = idx.loc[ idx['comid'] == comid ]['index'].values[0]
		self.xs = xs[ xs['COMID'] == str(comid)]
		self.get_handrc()
		if self.get_xs() == None: 
			raise TypeError

	def get_xs(self):
		xscurves = []
		for rs in self.xs['RiverStation'].unique():
			current = self.xs[ self.xs['RiverStation'] == rs ]
			prof = current['ProfileM'].unique()[0]
			disch = map(float,current['Discharge_cfs_'].values)
			stage = map(float,current['Stage_Height_ft_'].values)
			pack = (prof,zip(disch, stage))
			xscurves.append( pack )
		if len(xscurves) != 0:
			self.xscurves = xscurves
			return 1

	def get_stage(self,q):
		other_stages = [] 
		for i in self.xscurves:
			disch, stage = zip(*i[1])
			interp_other = scipy.interpolate.interp1d(disch,stage,kind='linear')
			other_stages.append( interp_other(q) )		
		self.interp_hand = scipy.interpolate.interp1d(self.handq,self.handh,kind='linear')
		hand_stage = self.interp_hand(q)
		return (hand_stage, other_stages)

	def plot_rc(self):
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots()

		# plot all linearly-interpolated XS rating curves
		for i in self.xscurves:
			disch, stage = zip(*i[1])
			f = scipy.interpolate.interp1d(disch,stage,kind='linear')
			ax.plot(disch,f(disch),c='grey', linewidth=2)#, label= i[0])

		# plot linearly-interpolated HAND rating curve
		f = scipy.interpolate.interp1d(self.handq,self.handh,kind='linear') # kind=cubic
		ax.plot(self.handq[:41],f(self.handq[:41]),
			label='hand',c='b', linewidth=5)#,s=400,marker='^'

		# plot settings
		fig.set_size_inches(20,16, forward=True)
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

	xsintersect = pandas.read_csv('/home/paul/Desktop/xsintersect.csv',
		usecols=['COMID','RiverStation','ProfileM'])
	xsintersect = xsintersect.astype(str)
	xsrating = pandas.read_csv('/home/paul/Desktop/xsrating.csv',
		usecols=['Stage_Height_ft_','Discharge_cfs_','RiverStation'])
	xsrating = xsrating.astype(str)
	xs = pandas.merge(xsrating,xsintersect,
		on='RiverStation')
	# print xsintersect[ xsintersect['RiverStation'] == '44717' ]
	# print xsrating[ xsrating['RiverStation'] == '44717' ]


	# print xs



	for i in range(len(idlookup)):
		# i = 4
		comid = idlookup['FLComID'][i]

		# print xs.loc['COMID']
		# xsi = xs[ xs['COMID'] == str(comid)] 

		# curves = []
		# for rs in xsi['RiverStation'].unique():
		# 	current = xsi[ xsi['RiverStation'] == rs ]
		# 	prof = current['ProfileM'].unique()[0]
		# 	stage = current['Stage_Height_ft_'].values
		# 	disch = current['Discharge_cfs_'].values
		# 	pack = (prof,zip(stage,disch))
		# 	curves.append( pack )
		# print curves

		# break

		# if len(xs) != 0: 
		# 	print xs
		# 	for rs in intersect['RiverStation'].values:
		# 		print xsrating [ xsrating['RiverStation'] == int(rs) ]
		try: 
			rcdist = RCDist(comid,ncdf,idx,xs)
		except TypeError: 
			print 'COMID {0} contains no cross-sections'.format(str(comid))
			continue

		rcdist.get_stage(5000)
		rcdist.plot_rc()
		
		# handrc = HandRC(comid,ncdf,idx)
		# # print handrc.nwm_stage('short_range')
		# handrc.plot_rc()





# 297, 984

# print handrc.nwm_stage(974)



# for s in series:
# 	dates = s['dates']
# 	for i, v in enumerate(s['values']):
# 		print dates[i].strftime('%y-%m-%d %H'), '\t', v