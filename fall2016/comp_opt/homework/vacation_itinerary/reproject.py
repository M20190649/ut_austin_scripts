from qgis.core import *
from qgis.gui import *
from PyQt4.QtCore import *
import glob
import os

# Supply path to QGIS install location
QgsApplication.setPrefixPath('/usr/bin/qgis',True)
# Create reference to QgsApplication
qgs = QgsApplication([],True) # No GUI
qgs.initQgis()

files = glob.glob('/media/paul/pman/compopt/roadnetwork/usa/raw/*.shp')
for file in files:
	path = os.path.basename(file)
	name = os.path.splitext(path)[0]
	state = name.split('_')[0]

	1/0
	if file == '/media/paul/pman/compopt/roadnetwork/usa/raw/alaska_roads.shp':
		path = os.path.basename(file)
		name = os.path.splitext(path)[0]
		# Load road layer
		layer = QgsVectorLayer(file,name,'ogr')
		QgsMapLayerRegistry.instance().addMapLayer(layer)

		# Set destination coordinate system
		crsDest = QgsCoordinateReferenceSystem(102010, QgsCoordinateReferenceSystem.EpsgCrsId)

		# Write to new shapefile
		output = '/media/paul/pman/compopt/roadnetwork/usa/reproj/' + name + '_reproj.shp'
		QgsVectorFileWriter.writeAsVectorFormat(
			layer,output,'utf-8',crsDest,'ESRI Shapefile')
		print '{0} finished!'.format(name + '_reproj.shp')

# Remove provider and layer registries from memory
qgs.exitQgis()