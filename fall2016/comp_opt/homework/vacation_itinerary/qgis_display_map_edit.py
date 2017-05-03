# **********************
# Display road network and nodes in qgis
# **********************
# from qgis.core import *
# from qgis.gui import *
# from PyQt4.QtCore import *

# # Supply path to QGIS install location
# QgsApplication.setPrefixPath('/usr/bin/qgis',True)
# # Create reference to QgsApplication
# qgs = QgsApplication([],True) # No GUI
# qgs.initQgis()

# # Initialize map canvas
# canvas = QgsMapCanvas()
# canvas.setCanvasColor(Qt.white)
# canvas.enableAntiAliasing(True) # Smooth rendering

# # Load road layer
# layer = QgsVectorLayer(edge_shp,'roadseg','ogr')
# QgsMapLayerRegistry.instance().addMapLayer(layer)
# canvas.setExtent(layer.extent())
# canvas.setLayerSet([QgsMapCanvasLayer(layer)])

# # Load attractions layer
# layer = QgsVectorLayer(edge_shp,'roadseg','ogr')
# QgsMapLayerRegistry.instance().addMapLayer(layer)
# canvas.setExtent(layer.extent())
# canvas.setLayerSet([QgsMapCanvasLayer(layer)])

# # Load hotel layer
# layer = QgsVectorLayer(edge_shp,'roadseg','ogr')
# QgsMapLayerRegistry.instance().addMapLayer(layer)
# canvas.setExtent(layer.extent())
# canvas.setLayerSet([QgsMapCanvasLayer(layer)])

# # Display map
# canvas.refresh()
# canvas.show()
# qgs.exec_()

# # Remove provider and layer registries from memory
# qgs.exitQgis()