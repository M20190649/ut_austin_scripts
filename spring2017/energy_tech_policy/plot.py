import pandas
import matplotlib.pyplot as plt
import matplotlib
import scipy

csvfile = 'us_summary_2010_clean_trimmed_edit.csv'

label='StateCode'

query1 = 'IEA_Residential_Energy_Consumption_Per_Capita_Million_BTU'

query2 = 'USGS_Public_Supply_Total_Water_Withdrawals_Per_Capita_GPD'

query3 = 'NOAA_Temperature_Annual_Average_Farenheit'

query4 = 'NOAA_Precipitation_Annual_Average_Inches'

df = pandas.read_csv(csvfile, usecols=[label,query1,query2,query3,query4])

# df = df[df.StateCode != 'AK'] # Remove Alaska

fig,ax = plt.subplots(figsize=(50,30),dpi=100)

matplotlib.rcParams.update({'font.size': 20}) # Change font size everywhere, mainly for point labels

bmin = 10
bmax = 60
step = 5

# Plot actual slopes
plt.scatter(x=df[query1],y=df[query2],c=df[query4],vmin=bmin,vmax=bmax, # 40,70 for temp
	s=1.13**df[query3],cmap='YlGnBu',label='{0}'.format(query3.replace('_',' ')))
plt.legend(loc='best',prop={'size':40})

cbar = plt.colorbar(ticks=scipy.arange(bmin,(bmax+step),step))
cbar.set_label(query4.replace('_',' '),size=40)
cbar.ax.tick_params(labelsize=32)

for xy,label in zip(zip(df[query1],df[query2]),df[label].unique()):
	ax.annotate('{0}'.format(label), xy=xy, textcoords='data')

plt.xlabel('{0}'.format('EIA Residential Energy Consumption Per Capita Million BTU'),size=40)
plt.ylabel('{0}'.format(query2.replace('_',' ')),size=40)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
# plt.xlim(0,1000)
# plt.ylim(0,12000)
# fig.tight_layout()
# ax.get_xaxis().get_major_formatter().set_scientific(False) # Remove scientific notation on x-axis
plt.title(
	'Per Capita Residential Energy Consumption\nvs. Per Capita Public Supply Water Withdrawals\nvs. Average Temperature vs. Average Precipitation',
	size=60,y=1.01)
plt.grid(True)

# plt.show()
plt.savefig('residential_energy_vs_publicsupply_water_vs_avg_temp_and_precip.png')