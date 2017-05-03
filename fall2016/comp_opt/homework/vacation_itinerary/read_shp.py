from osgeo import gdal, ogr

class Shapefile:

	def __init__(self,shp,driver='ESRI Shapefile'):
		self.shp = shp
		self.driver = ogr.GetDriverByName(driver)
		self.source = self.driver.Open(shp,0)
		self.layer = self.source.GetLayer()

	def field_names(self):
		layerdef = self.layer.GetLayerDefn()
		fields = [layerdef.GetFieldDefn(i).GetName() 
			for i in range(layerdef.GetFieldCount())]
		return fields

	def read_shp(self,fields):
		for field in fields:
			print field
			print self.layer.GetField('fclass')
		# for field in fields:
		# 	for item in self.layer:
		# 		print item.GetField('name')

if __name__ == '__main__':
	catchmentshp = '/media/paul/pman/compopt/roadnetwork/shapefiles/texas.shp'
	catch = Shapefile(catchmentshp)
	fields = catch.field_names()
	print fields
	# catch.read_shp(fields)