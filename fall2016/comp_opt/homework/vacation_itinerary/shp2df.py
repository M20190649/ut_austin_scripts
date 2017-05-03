import scipy
import pandas

def shp2df(fname,kind='qgis',fields=False): 
	"""Convert from ESRI shapefile to Pandas dataframe
	from: http://permalink.gmane.org/gmane.comp.python.pydata/1870"""
	if kind == 'shapefile':
		import shapefile # https://github.com/GeospatialPython/pyshp
		shp = shapefile.Reader(fname)
		records = shp.records() # records (rows)
		# shapes = list(shp.iterShapes())
		# print shapes
		fld = scipy.array(shp.fields[1:],dtype=str)
		data = pandas.DataFrame(records, columns=fld[:,0])
		del shp
		if fields:
			data = data[[f for f in fields]]
		return data

	if kind == 'ogr':
		from osgeo import ogr,osr
		driver = ogr.GetDriverByName('ESRI Shapefile')
		shp = driver.Open(fname,0)
		layer = shp.GetLayer()
		# spatialref = layer.GetSpatialRef()
		# layerdef = layer.GetLayerDefn()
		for i in range(layer.GetFeatureCount()):
			feature = layer.GetFeature(i)
			print feature.GetField('fclass')

	if kind == 'geopandas':
		import geopandas
		gdf = geopandas.GeoDataFrame.from_file(fname)
		print gdf

	if kind == 'qgis':
		import qgis
		from PyQt4.QtCore import QVariant
		# qgis.core.QgsApplication.setPrefixPath('/usr/bin/qgis',False)
		qgs = qgis.core.QgsApplication(['usr/bin/qgis'],False) # False = no GUI
		qgs.initQgis()
		layer = qgis.core.QgsVectorLayer(fname,'road','ogr')
		if fields:
			temp = qgis.core.QgsVectorLayer('LineString?crs=epsg:4326','road','memory')
			pr = temp.dataProvider()
			temp.startEditing()
			for field in fields:
				pr.addAttributes([
					qgis.core.QgsField(field,QVariant.String)
					])
			for feature in layer.getFeatures():
				newFeature = qgis.core.QgsFeature()
				newFeature.setGeometry(feature.geometry())
				for field in fields:
					newFeature.setAttributes([feature.attribute(field)])
				pr.addFeatures([newFeature])

			temp.commitChanges()
			temp.updateExtents()
			layer = qgis.core.QgsMapLayerRegistry.instance().addMapLayer(temp) # maplayer

		if not layer.isValid(): # Check if layer loaded correctly
			print 'Layer failed to load'
			print len(layer.pendingFields())
			return
		print 'Layer loaded successfully!'

		# crs = qgis.core.QgsCoordinateReferenceSystem(4326, 
		#	qgis.core.QgsCoordinateReferenceSystem.PostgisCrsId) # or InternalCrsId or EpsgCrsId
		# layer.setCrs(crs)

		# fieldnames = [field.name() for field in layer.pendingFields()]
		# print fieldnames

		# (vectorFileName, theFileEncoding, fileEncoding, fields, geometryType, 
		# srs, driverName, datasourceOptions, layerOptions, newFilename, 
		# fieldValueConverter, layerName, action)
		qgis.core.QgsVectorFileWriter.writeAsVectorFormat(layer,'test.csv',
			'utf-8',None,'CSV',layerOptions='GEOMETRY=AS_XYZ')

		qgs.exitQgis()

def get_geometry(layer):
	for f in layer.getFeatures():
		geom = f.geometry()
		print 'Feature ID: {0}'.format(f.id())
		if geom.type() == qgis.core.QGis.Point:
			x = geom.asPoint()
			print 'Point: ' + str(x)
		elif geom.type() == qgis.core.QGis.Line:
			x = geom.asPolyline()
			print 'Line: ' + str(x)
		elif geom.type() == qgis.core.QGis.Polygon:
			x = geom.asPolygon()
			print 'Polygon: ' + str(x)
		else:
			print 'Unknown'	

if __name__ == '__main__':
	roadnw = 'texas/texas.shp'
	df = shp2df(roadnw,kind='shapefile',
		fields=['oneway','LENGTH_GEO','START_X','START_Y','END_X','END_Y'])
	print df