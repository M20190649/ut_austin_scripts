import pandas
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.optimize import curve_fit

csvfile = 'us_summary_2010_final_clean.csv'

label='StateCode'

pop = 'USGS_Population'

# Per Capita
query1 = 'IEA_Residential_Energy_Consumption_Per_Capita_Million_BTU' 
query2 = 'USGS_Public_Supply_Total_Water_Withdrawals_Per_Capita_GPD'
divisor = 1e-6

# # NOT Per Capita
# query1 = 'IEA_Residential_Energy_Consumption_Billion_BTU'
# query2 = 'USGS_Public_Supply_Total_Water_Withdrawals_MGD'
# divisor = 1e-3

query3 = 'NOAA_Temperature_Annual_Average_Farenheit'

query4 = 'NOAA_Precipitation_Annual_Average_Inches'

df = pandas.read_csv(csvfile, usecols=[label,pop,query1,query2,query3,query4])

# df = df[df.StateCode != 'AK'] # Remove Alaska

fig,ax = plt.subplots(figsize=(50,30),dpi=100)

matplotlib.rcParams.update({'font.size': 20}) # Change font size everywhere, mainly for point labels



iea_over_usgs = (df[query2]/df[query1])/divisor/df[pop]

idx = iea_over_usgs.argsort()
x = iea_over_usgs[idx].values
tmp = df[query3][idx].values
pcp = df[query4][idx].values

# for i in range(len(x)):
# 	print x[i], tmp[i], pcp[i]

# Plot actual slopes
plt.scatter(x=x,y=tmp,s=400,c='red',label='USGS/EIA Ratio (GPD/BTU) vs. Temperature (F)')
plt.scatter(x=x,y=pcp,s=400,c='blue',label='USGS/EIA Ratio (GPD/BTU) vs. Precipitation (In.)')
plt.legend(loc='best',prop={'size':40})

def f(x, a, b, c):
	return a * scipy.exp(b * x) + c

tmp_opt, tmp_cov = curve_fit(f, x, tmp, p0=(1e-6,1e-6,1))
pcp_opt, pcp_cov = curve_fit(f, x, pcp, p0=(1e-6,1e-6,1))

print tmp_opt, tmp_cov
print pcp_opt, pcp_cov

# Retrieve R-squared value
def get_r_squared(x,y,opt):
	residuals = y - f(x, *opt)
	ss_res = scipy.sum( residuals**2 )
	ss_tot = scipy.sum( ( y - scipy.mean(y) )**2 )
	r_squared = 1 - (ss_res / ss_tot)
	return r_squared

rs_tmp = get_r_squared(x,tmp,tmp_opt)
rs_pcp = get_r_squared(x,pcp,pcp_opt)
print rs_tmp
print rs_pcp

plt.text(0.83,0.48,'R$^2$: {0:.3}'.format(rs_tmp),ha='center', va='center', transform=ax.transAxes)
plt.text(0.83,0.18,'R$^2$: {0:.3}'.format(rs_pcp),ha='center', va='center', transform=ax.transAxes)

plt.plot(x, f(x, *tmp_opt), c = 'red')
plt.plot(x, f(x, *pcp_opt), c = 'blue')

# cbar = plt.colorbar(ticks=scipy.arange(bmin,(bmax+step),step))
# cbar.set_label(query4.replace('_',' '),size=40)
# cbar.ax.tick_params(labelsize=32)

# for xy,label in zip(zip(df[query1],df[query2]),df[label].unique()):
# 	ax.annotate('{0}'.format(label), xy=xy, textcoords='data')

plt.xlabel('{0}'.format('USGS Public Supply over EIA Residential, per Capita (BTU/GPD)'),size=40)
plt.ylabel('{0}'.format('NOAA Annual Average Temperature (F) and Precipitation (In.)'),size=40)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.xlim(x[0]-.1,x[-1]+.1)

# plt.gca().set_yscale('log')

# plt.gca().set_xscale('log')
# plt.gca().axis('tight')
# plt.xlim(0,1000)
# plt.ylim(0,12000)
# fig.tight_layout()
# ax.get_xaxis().get_major_formatter().set_scientific(False) # Remove scientific notation on x-axis
plt.title(
	'Public Supply Water Withdrawals over Residential Energy Consumption\nvs. Average Temperature',
	size=60,y=1.01)
plt.grid(True)

# plt.show()
plt.savefig('usgs_eia_ratio_percap_vs_avg_temp_and_precip_trendlines.png')