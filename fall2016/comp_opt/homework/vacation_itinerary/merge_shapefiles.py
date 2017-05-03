# Merge a bunch of shapefiles with attributes quickly!
import glob
import shapefile
files = glob.glob("/media/paul/pman/compopt/roadnetwork/osmfiles/usa/*.shp")
w = shapefile.Writer()
for f in files:
  r = shapefile.Reader(f)
  w._shapes.extend(r.shapes())
  w.records.extend(r.records())
w.fields = list(r.fields)
w.save("merged")