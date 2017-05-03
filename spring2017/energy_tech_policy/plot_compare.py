import pandas
import matplotlib.pyplot as plt
import matplotlib
import scipy

csvfile = 'us_summary_2010_final_clean.csv'

label='StateCode'

bar1 = 'IEA_Total_Energy_Consumed_Per_Capita_Million_BTU' # 'IEA_Total_Energy_Consumed_BTU'

bar2 = 'USGS_Total_Water_Withdrawals_Per_Capita_GPD' # 'USGS_Total_Water_Withdrawals_GPD'

line1 = 'NOAA_Temperature_Annual_Average_Farenheit'

line2 = 'NOAA_Precipitation_Annual_Average_Inches'

df = pandas.read_csv(csvfile)#, usecols=[label,bar1,bar2,line1,line2])

print df.columns.values.tolist()
1/0
fig,ax = plt.subplots(figsize=(50,30),dpi=100)

matplotlib.rcParams.update({'font.size': 20}) # Change font size everywhere, mainly for point labels

bmin = 10
bmax = 60
step = 5

# Plot actual slopes
plt.scatter(x=df[bar1],y=df[bar2],c=df[line2],vmin=bmin,vmax=bmax, # 40,70 for temp
	s=1.13**df[line1],cmap='YlGnBu',label='{0}'.format(line1.replace('_',' ')))
plt.legend(loc='best',prop={'size':40})

cbar = plt.colorbar(ticks=scipy.arange(bmin,(bmax+step),step))
cbar.set_label(line2.replace('_',' '),size=40)
cbar.ax.tick_params(labelsize=32)

for xy,label in zip(zip(df[bar1],df[bar2]),df[label].unique()):
	ax.annotate('{0}'.format(label), xy=xy, textcoords='data')

plt.xlabel('{0}'.format(bar1.replace('_',' ')),size=40)
plt.ylabel('{0}'.format(bar2.replace('_',' ')),size=40)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlim(0,1000)
plt.ylim(0,12000)
# fig.tight_layout()
# ax.get_xaxis().get_major_formatter().set_scientific(False) # Remove scientific notation on x-axis
plt.title(
	'Per Capita Total Energy Consumption\nvs. Per Capita Total Water Withdrawals\nvs. Average Temperature vs. Average Precipitation',
	size=60,y=1.01)
plt.grid(True)

# plt.show()
plt.savefig('total_energy_vs_total_water_vs_avg_temp_and_precip.png')