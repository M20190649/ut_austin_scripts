from netCDF4 import Dataset
import itertools
import csv
import scipy

class readNCDF:

	def __init__(self,ncdf_path):
		""" Read ncdf file """
		self.ncdf = Dataset(ncdf_path,mode='r')

		print self.ncdf.variables.keys()

		for i in range(len(self.ncdf.variables['COMID'])):
			self.get_vals(i)
			# idx = self.ncdf.variables['NEWID'][i]
			# filename = 'ncdf_{0}.csv'.format(idx)
			# self.write_csv(fname=filename)
		self.ncdf.close()

	def get_vals(self,idx):
		self.handq = self.ncdf.variables['Q_cfs']
		self.handh = self.ncdf.variables['H_ft']
		print self.handq
		print self.handh

	# def write_csv(self,fname):
	# 	rows = itertools.izip_longest(self.stage,self.width,
	# 		self.sarea,self.warea,self.wperim,self.barea,
	# 		self.hydrad,self.vol,self.disch,fillvalue='')
	# 	with open('ncdf_data/{0}'.format(fname), 'w') as f:
	# 		writer = csv.writer(f)
	# 		writer.writerow(['COMID',self.comid])
	# 		writer.writerow('')
	# 		writer.writerow([item for item in self.ncdf.variables]) # headers
	# 		writer.writerows(rows)
	# 	f.close()

if __name__ == '__main__':
	ncdf_path = 'oniondata/handratingcurves.nc'
	readNCDF(ncdf_path)