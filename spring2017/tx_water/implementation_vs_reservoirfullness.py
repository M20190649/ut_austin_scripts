import pandas
import matplotlib.pyplot as plt
import matplotlib

year = '2013'

pfull = pandas.read_csv('reservoir_levels/reservoir_annual_pfull_summary.csv')

impl = pandas.read_csv('tx_water_plan_implementation_summary_edit.csv',usecols=['Region','Complete_Percentage'])

impl = impl[impl['Region'] != 'E'] # get rid of e

# pfull = pfull[pfull['Region'] != 'P'] # get rid of p
# impl = impl[impl['Region'] != 'P'] # get rid of p



x = impl['Complete_Percentage']

avg_vals = pfull[['2011','2012','2013','2014','2015','2016','2017']].mean(axis=1).values
y = avg_vals

matplotlib.rcParams.update({'font.size':12}) # Change font size everywhere, mainly for point labels

for xy,label in zip(zip(x,y),impl['Region']):
	plt.annotate('{0}'.format(label), xy=xy, textcoords='data')

plt.title('Water Management Strategy Implementation (2016)\nvs. Average Reservoir Fullness (2011-2017)',size=20)
plt.xlabel('Percent Implementation',size=16)
plt.ylabel('Reservoir Percent Full',size=16)
plt.xlim(0,100)
plt.ylim(0,100)
plt.scatter(x,y,s=400,c='lightgreen')
plt.grid(True)
plt.tight_layout()
plt.savefig('implementation_vs_reservoir_fullness.png')
# plt.show()