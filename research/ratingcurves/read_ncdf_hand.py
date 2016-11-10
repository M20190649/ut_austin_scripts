from netCDF4 import Dataset
import itertools
import csv
import scipy

class readNCDF:

	def __init__(self,ncdf_path,upto=83):
		""" Read ncdf file """
		self.ncdf = Dataset(ncdf_path,mode='r')

		for i in range(len(self.ncdf.variables['NEWID'][:])):
			self.get_vals(i,upto=upto)
			idx = self.ncdf.variables['NEWID'][i]
			filename = 'ncdf_{0}.csv'.format(idx)
			self.write_csv(fname=filename)
		self.ncdf.close()

	def get_vals(self,idx,upto):
		stage = self.ncdf.variables['StageHeight'][:upto]
		stage = scipy.rint(stage*3.28084) # Convert m to ft and round to int
		self.stage = stage
		self.newid = self.ncdf.variables['NEWID'][:upto][idx]
		self.length = self.ncdf.variables['Length'][:upto][idx]
		self.slope = self.ncdf.variables['Slope'][:upto][idx]
		self.rough = self.ncdf.variables['Roughness'][:upto][idx]
		self.width = self.ncdf.variables['Width'][:upto][idx]
		self.sarea = self.ncdf.variables['SurfaceArea'][:upto][idx]
		self.warea = self.ncdf.variables['WetArea'][:upto][idx]
		self.wperim = self.ncdf.variables['WettedPerimeter'][:upto][idx]
		self.barea = self.ncdf.variables['BedArea'][:upto][idx]
		self.hydrad = self.ncdf.variables['HydraulicRadius'][:upto][idx]
		self.vol = self.ncdf.variables['Volume'][:upto][idx]
		self.disch = self.ncdf.variables['Discharge'][:upto][idx]
		
	def write_csv(self,fname):
		rows = itertools.izip_longest(self.stage,self.width,
			self.sarea,self.warea,self.wperim,self.barea,
			self.hydrad,self.vol,self.disch,fillvalue='')
		with open('ncdf_data/{0}'.format(fname), 'w') as f:
			writer = csv.writer(f)
			writer.writerow(['NEWID',self.newid])
			writer.writerow(['LENGTH_KM',self.length])
			writer.writerow(['SLOPE',self.slope])
			writer.writerow(['ROUGHNESS',self.rough])
			writer.writerow('')
			writer.writerow([item for item in self.ncdf.variables]) # headers
			writer.writerows(rows)
		f.close()

if __name__ == '__main__':
	ncdf_path = 'oniondata/slaughter_hydraulic_properties.nc'
	readNCDF(ncdf_path)