#title           :hw2.py
#description     :HW 2 for ORI 397
#author          :Paul J. Ruess
#date            :20160912
#==============================================================================

import pylab
import geoplotter
import pandas
import matplotlib

class MilexPlotter:
	"""Creates MilexPlotter instance map"""

	def __init__(self, csv, shp, name):
		self.g = geoplotter.GeoPlotter()
		self.g.readShapefile(shp, name)
		self.df = pandas.read_csv(csv)

	def getCinc(self, year, cow):
		cincdf = self.df[(self.df.year == year) & (self.df.ccode == cow)]
		try:
			cinc = cincdf['cinc'].values[0]
		except IndexError:
			cinc = False
		return cinc

	def getCows(self):
		cowsdf = self.df.drop_duplicates(['ccode'])
		return cowsdf.ccode

	def plotCountry(self, cow, **kwargs):
		indices = []
		for i in range(len(self.g.m.countries_info)):
			if self.g.m.countries_info[i]['COWCODE'] == cow: 
				indices.append(i)
		self.g.drawShapes('countries', indices, **kwargs)

	def plotYear(self, year):
		self.g.clear()
		self.g.drawWorld('blue')
		maxCINC = self.df.cinc.max()
		norm = matplotlib.colors.Normalize(0,maxCINC)
		cm = matplotlib.cm
		for cow in self.getCows():
			if self.getCinc(year, cow):
				cinc = self.getCinc(year, cow)
				self.plotCountry(cow, facecolor=cm.ScalarMappable(norm, cm.hot).to_rgba(cinc))
				self.g.figureText(-150,-30,year,fontsize=24,fontweight='bold')

	def plotAllYears(self):
		for year in self.df.drop_duplicates(['year']).year:
			self.plotYear(year)
			self.g.savefig('plots/cinc_{0}.png'.format(year))

if __name__ == "__main__":

	csv = 'NMC_v4_0.csv'
	shp = 'cshapes_0.4-2/cshapes'
	name = 'countries'

	MP = MilexPlotter(csv, shp, name)
	MP.plotAllYears()
	pylab.show()