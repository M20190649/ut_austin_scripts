#!/bin/bash/

# mkdir merged
# for f in (*.shp) do (
# if not exist merged/merged.shp (
# ogr2ogr -f “esri shapefile” merged/merged.shp %f) else (
# ogr2ogr -f “esri shapefile” -update -append merged/merged.shp %f -nln Merged )
# )

# for f in *.shp; do ogr2ogr -update -append merge.shp $f -f "ESRI Shapefile"; done;

# DATA=`find . -name '*.shp'`
# echo $DATA
# ogr2ogr -a_srs EPSG:4326 merge.shp
# for i in $DATA
# do
# ogr2ogr -append -update merge.shp $i -f "Esri Shapefile"
# done
