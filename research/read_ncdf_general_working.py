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
			filename = 'ncdf_{0}.csv'.format(str(i))
			self.write_csv(fname=filename)

	def get_vals(self,idx,upto):
		stage = self.ncdf.variables['StageHeight'][:upto]
		stage = scipy.rint(stage*3.28084) # Convert m to ft and round to int
		self.stage = stage[idx]
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
		
	def write_to_csv(self,fname):
		for i,j in zip( range(len(self.items)), range(self.items[0][-1]) ):
			print self.items[i][0]

		for i in range(len(ncdf.variables['NEWID'][:])):

			label = ncdf.variables['NEWID'][i]

			with open('ncdf_data/ncdf_{0}.csv'.format(label), 'w') as f:
				writer = csv.writer(f)
				writer.writerow([j.name for j in self.items]) # headers
				writer.writerow(rows)
				# writer.writerows(rows)
			f.close()

		# print items[5][:]
		# for i in range(len(self.items)):
		# self.write_to_csv(self.items)
			# print items[i].units
			# print items[i][:]
			# if len(items[i][:])==(len(items)):
				# print items[i][:]
		# for i,x in enumerate(items): 
		# 	print x, x[i]
		# for var in ncdf.variables: 
		# 	varnames.append(var) # create list of available data
		# 	item = ncdf.variables[var]
		# 	# print item.name, item[0]
		# 	for i in item:
		# 		print i

		ncdf.close()



		# info = [self.comid,self.usgsid,self.handslope]
		# rows = itertools.izip_longest(self.items)
		# with open('onionck/results/newcsvdata/{0}_data.csv'.format(self.comid), 'w') as f:
		# 	csv.writer(f).writerows(rows)
		# f.close()


if __name__ == '__main__':
	ncdf_path = 'oniondata/slaughter_hydraulic_properties.nc'
	nc = readNCDF(ncdf_path)

# ['StageHeight', 'NEWID', 'Length', 'Slope', 'Roughness', 'Width', 
# 'SurfaceArea', 'WetArea', 'WettedPerimeter', 'BedArea', 'HydraulicRadius', 
# 'Volume', 'Discharge']
		# stage, newid, length, slope, rough, width, sarea, warea, wetperim, barea, hydrorad, vol, disch = [ncdf.variables[v] for v in ncdf.variables]
		# for i in range(len(ncdf.variables)):
		# 	print stage[i]
		# 	print newid[i]
		# 	print length[i]
		# 	print slope[i]
		# 	print rough[i]
		# 	print width
		# 	print sarea
		# 	print warea
		# 	print wetperim
		# 	print barea
		# 	print hydrorad
		# 	print vol
		# 	print disch