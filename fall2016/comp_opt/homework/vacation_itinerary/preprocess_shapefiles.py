from qgis.core import *
from qgis.gui import *
from PyQt4.QtCore import *
import glob
import os


dest = 'texas'

# Supply path to QGIS install location
QgsApplication.setPrefixPath('/usr/bin/qgis',True)
# Create reference to QgsApplication
qgs = QgsApplication([],True) # No GUI
qgs.initQgis()

raws = glob.glob('/media/paul/pman/compopt/roadnetwork/usa/raw/*.shp')
for raw in raws:

	if raw == '/media/paul/pman/compopt/roadnetwork/usa/raw/{0}_roads.shp'.format(dest):

		print '{0} started...'.format(dest)
		
		# Load road layer
		path = os.path.basename(raw)
		name = os.path.splitext(path)[0]
		state = name.split('_')[0]
		layer = QgsVectorLayer(raw,name,'ogr')
		# QgsMapLayerRegistry.instance().addMapLayer(layer)

		# # Create selection expression
		expr = QgsExpression(" \
			fclass = 'livingstreet' OR \
			fclass = 'motorway' OR \
			fclass = 'motorway_link' OR \
			fclass = 'primary' OR \
			fclass = 'primary_link' OR \
			fclass = 'residential' OR \
			fclass = 'secondary' OR \
			fclass = 'secondary_link' OR \
			fclass = 'service' OR \
			fclass = 'tertiary' OR \
			fclass = 'tertiary_link' OR \
			fclass = 'trunk' OR \
			fclass = 'trunk_link' OR \
			fclass = 'unclassified' \
			")

		# # Retrieve selection and overwrite layer with selected features
		selections = layer.getFeatures( QgsFeatureRequest( expr ) )
		ids = [s.id() for s in selections]
		layer.setSelectedFeatures( ids )

		print 'Features selected...'

		# Load data provider
		geom_pr = layer.dataProvider()

		# Add features to layer
		layer.startEditing()

		caps = layer.dataProvider().capabilities()

		if caps & QgsVectorDataProvider.ChangeAttributeValues:

			print 'Attribute manipulation initiated...'

			# d = QgsDistanceArea()
			# d.setEllipsoid('NAD83') # WGS84
			# d.setEllipsoidalMode(True)

			# Create necessary fields
			geom_pr.addAttributes([
				QgsField('START_Y', QVariant.Double),
				QgsField('START_X', QVariant.Double),
				QgsField('END_Y', QVariant.Double),
				QgsField('END_X', QVariant.Double)
				])

			print 'Attribute fields added...'

			for feature in layer.selectedFeatures():
				# Retrieve line geometries
				geom = feature.geometry().asPolyline()
				
				# Retrieve start and end node latlon values
				slat = geom[0][1] # start latitude
				slon = geom[0][0] # start longitude
				elat = geom[-1][1] # end latitude
				elon = geom[-1][0] # end longitude

				# Define attributes to add to attribute table
				geom_attrs = {}
				geom_attrs[geom_pr.fieldNameMap()['START_Y']] = slat
				geom_attrs[geom_pr.fieldNameMap()['START_X']] = slon
				geom_attrs[geom_pr.fieldNameMap()['END_Y']] = elat
				geom_attrs[geom_pr.fieldNameMap()['END_X']] = elon
				
				# Add attributes specified by attr dict to attribute table
				geom_pr.changeAttributeValues({feature.id() : geom_attrs})

			print 'All feature attributes added...'

		# Commit changes
		layer.commitChanges()

		# Write to output
		output = '/media/paul/pman/compopt/roadnetwork/usa/latlon/' + state + '_roads_latlon.shp'
		QgsVectorFileWriter.writeAsVectorFormat(
			layer,output,'UTF8',layer.crs(),'ESRI Shapefile',1)
		print '{0} finished!'.format(state + '_roads_latlon.shp')

latlons = glob.glob('/media/paul/pman/compopt/roadnetwork/usa/latlon/*.shp')
for latlon in latlons:
	if latlon == '/media/paul/pman/compopt/roadnetwork/usa/latlon/{0}_roads_latlon.shp'.format(dest):
		path = os.path.basename(latlon)
		name = os.path.splitext(path)[0]
		state = name.split('_')[0]
		
		# Load road layer
		layer = QgsVectorLayer(latlon,name,'ogr')
		# QgsMapLayerRegistry.instance().addMapLayer(layer)

		# Set destination coordinate system
		crsDest = QgsCoordinateReferenceSystem(102010, QgsCoordinateReferenceSystem.EpsgCrsId)

		# Write to new shapefile
		output = '/media/paul/pman/compopt/roadnetwork/usa/reproj/' + state + '_roads_reproj.shp'
		QgsVectorFileWriter.writeAsVectorFormat(
			layer,output,'utf-8',crsDest,'ESRI Shapefile')
		print '{0} finished!'.format(state + '_roads_reproj.shp')

reprojs = glob.glob('/media/paul/pman/compopt/roadnetwork/usa/reproj/*.shp')
for reproj in reprojs:

	if reproj == '/media/paul/pman/compopt/roadnetwork/usa/reproj/{0}_roads_reproj.shp'.format(dest):

		print '{0} started...'.format(dest)
		
		# Load road layer
		path = os.path.basename(reproj)
		name = os.path.splitext(path)[0]
		state = name.split('_')[0]

		layer = QgsVectorLayer(reproj,name,'ogr')
		# QgsMapLayerRegistry.instance().addMapLayer(layer)

		# Load data provider
		len_pr = layer.dataProvider()

		# Add features to layer
		layer.startEditing()

		caps = layer.dataProvider().capabilities()

		if caps & QgsVectorDataProvider.ChangeAttributeValues:

			print 'Attribute manipulation initiated...'

			# d = QgsDistanceArea()
			# d.setEllipsoid('NAD83') # WGS84
			# d.setEllipsoidalMode(True)

			# Create necessary fields
			len_pr.addAttributes([
				QgsField('LENGTH_GEO', QVariant.Double)
				])

			print 'Attribute field added...'

			for feature in layer.getFeatures():
				# Retrieve line geometries
				geom = feature.geometry().asPolyline()

				# Retrieve geodesic length
				l = feature.geometry().length() / 1609.34 # convert to miles

				# Define attributes to add to attribute table
				len_attrs = {}
				len_attrs[len_pr.fieldNameMap()['LENGTH_GEO']] = l
				
				# Add attributes specified by attr dict to attribute table
				len_pr.changeAttributeValues({feature.id() : len_attrs})

			print 'All feature attributes added...'

		# Commit changes
		layer.commitChanges()

		# Write to output
		output = '/media/paul/pman/compopt/roadnetwork/usa/length/' + state + '_roads_length.shp'
		QgsVectorFileWriter.writeAsVectorFormat(
			layer,output,'UTF8',layer.crs(),'ESRI Shapefile')
		print '{0} finished!'.format(state + '_roads_length.shp')

lengths = glob.glob('/media/paul/pman/compopt/roadnetwork/usa/length/*.shp')
for length in lengths:
	if length == '/media/paul/pman/compopt/roadnetwork/usa/length/{0}_roads_length.shp'.format(dest):
		path = os.path.basename(length)
		name = os.path.splitext(path)[0]
		state = name.split('_')[0]
		
		# Load road layer
		layer = QgsVectorLayer(length,name,'ogr')
		# QgsMapLayerRegistry.instance().addMapLayer(layer)

		# Set destination coordinate system
		crsDest = QgsCoordinateReferenceSystem(4326, QgsCoordinateReferenceSystem.EpsgCrsId)

		# Write to new shapefile
		output = '/media/paul/pman/compopt/roadnetwork/usa/final/' + state + '_roads_final.shp'
		QgsVectorFileWriter.writeAsVectorFormat(
			layer,output,'utf-8',crsDest,'ESRI Shapefile')
		print '{0} finished!'.format(state + '_roads_final.shp')

# Remove provider and layer registries from memory
qgs.exitQgis()