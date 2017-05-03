import glob
import os
import shutil

files = glob.glob(
	'/media/paul/pman/compopt/roadnetwork/osmfiles/**/gis.osm_roads_free_1.*')
dest = '/media/paul/pman/compopt/roadnetwork/osmfiles/usa/'
for file in files:
	parent = os.path.dirname(file).split('/')[-1]
	ext = os.path.splitext(file)[1]
	filename = parent + '_roads' + ext
	print '{0} copied to {1}'.format(file,dest + filename)
	shutil.copyfile(file,dest + filename)