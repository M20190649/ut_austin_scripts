from qgis.core import *
from qgis.gui import *
from PyQt4.QtCore import *
import glob
import os

files = glob.glob('/media/paul/pman/compopt/roadnetwork/usa/reproj/*.shp')

# Supply path to QGIS install location
QgsApplication.setPrefixPath('/usr/bin/qgis',True)
# Create reference to QgsApplication
qgs = QgsApplication([],True) # No GUI
qgs.initQgis()

for file in files:

	if file == '/media/paul/pman/compopt/roadnetwork/usa/reproj/alaska_roads_reproj.shp':

		print 'Alaska started...'
		
		# Load road layer
		path = os.path.basename(file)
		name = os.path.splitext(path)[0]
		layer = QgsVectorLayer(file,name,'ogr')
		QgsMapLayerRegistry.instance().addMapLayer(layer)

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
		pr = layer.dataProvider()

		# Add features to layer
		layer.startEditing()

		caps = layer.dataProvider().capabilities()

		if caps & QgsVectorDataProvider.ChangeAttributeValues:

			print 'Attribute manipulation initiated...'

			# d = QgsDistanceArea()
			# d.setEllipsoid('NAD83') # WGS84
			# d.setEllipsoidalMode(True)

			# Create necessary fields
			pr.addAttributes([
				QgsField('START_Y', QVariant.Double),
				QgsField('START_X', QVariant.Double),
				QgsField('END_Y', QVariant.Double),
				QgsField('END_X', QVariant.Double),
				QgsField('LENGTH_GEO', QVariant.Double)
				])

			print 'Attribute fields added...'

			for feature in layer.getFeatures():
				# Retrieve line geometries
				geom = feature.geometry().asPolyline()
				
				# Retrieve start and end node latlon values
				slat = geom[0][1] # start latitude
				slon = geom[0][0] # start longitude
				elat = geom[1][1] # end latitude
				elon = geom[1][0] # end longitude

				# Retrieve geodesic length
				length = feature.geometry().length() / 1609.34 # convert to miles

				# Define attributes to add to attribute table
				attr = {}
				attr[pr.fieldNameMap()['START_Y']] = slat
				attr[pr.fieldNameMap()['START_X']] = slon
				attr[pr.fieldNameMap()['END_Y']] = elat
				attr[pr.fieldNameMap()['END_X']] = elon
				attr[pr.fieldNameMap()['LENGTH_GEO']] = length
				
				# Add attributes specified by attr dict to attribute table
				pr.changeAttributeValues({feature.id() : attr})

			print 'All feature attributes added...'

		# Commit changes
		layer.commitChanges()

		# Write to output
		output = '/media/paul/pman/compopt/roadnetwork/usa/final/' + name + '_final.shp'
		QgsVectorFileWriter.writeAsVectorFormat(
			layer,output,'UTF8',layer.crs(),'ESRI Shapefile',1)
		print '{0} finished!'.format(name + '_final_test.shp')

# Remove provider and layer registries from memory
qgs.exitQgis()