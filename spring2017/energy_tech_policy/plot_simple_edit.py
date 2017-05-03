import pandas
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy.optimize import curve_fit

# Create plot
def make_plot(x,y,c='b',s=400,fname=False,title='Exponential Fit of Raw Data',annotations=False,
	xtitle='X Axis',ytitle='Y Axis',xlog=False,ylog=False):
	# Initialize plotting instance
	fig,ax = plt.subplots(figsize=(50,30),dpi=100)

	# Organize data least to greatest
	idx = x.argsort()
	xdata = x[idx]
	ydata = y[idx]

	# Add annotations
	if list(annotations):
		labels = annotations[idx]
		matplotlib.rcParams.update({'font.size': 20}) # Change font size everywhere, mainly for point labels
		for xy,label in zip(zip(xdata,ydata),labels): 
			ax.annotate('{0}'.format(label), xy=xy, textcoords='data')

	# Line of best fit function, exponential
	def f(x, a, b, c):
		return a * scipy.exp(b*x) + c

	# Find best fit constants
	popt, pcov = curve_fit(f, xdata, ydata, p0=(1e-6,1e-6,1))
	print popt
	print pcov

	# Retrieve R-squared value
	residuals = ydata - f(xdata, *popt)
	ss_res = scipy.sum( residuals**2 )
	ss_tot = scipy.sum( ( ydata - scipy.mean(ydata) )**2 )
	r_squared = 1 - (ss_res / ss_tot)
	print r_squared

	# Plot data
	plt.scatter(x=xdata, y=ydata, s=s, c=c, label='Raw Data') # raw data
	plt.legend(loc='best',prop={'size':40})
	plt.plot(xdata, f(xdata, *popt), c=c, label='Exponential Fit')

	plt.xlabel('{0}'.format(xtitle),size=40)
	plt.ylabel('{0}'.format(ytitle),size=40)
	plt.xticks(fontsize=32)
	plt.yticks(fontsize=32)
	# plt.xlim(x[0],x[-1])

	if xlog: plt.gca().set_xscale('log')
	if ylog: plt.gca().set_yscale('log')

	# plt.xlim(0,1000)
	# plt.ylim(0,12000)
	# fig.tight_layout()
	plt.title(title,size=60,y=1.01)
	plt.grid(True)

	if fname: plt.savefig(fname)
	else: plt.show()

if __name__ == '__main__':
	csvfile = 'us_summary_2010_final_clean.csv'

	statenames='StateCode'
	energy = 'IEA_Residential_Energy_Consumption_Per_Capita_Million_BTU' 
	water = 'USGS_Public_Supply_Total_Water_Withdrawals_Per_Capita_GPD'
	temp = 'NOAA_Temperature_Annual_Average_Farenheit'
	precip = 'NOAA_Precipitation_Annual_Average_Inches'
	# iea_pop = 'IEA_Population'
	usgs_pop = 'USGS_Population'

	# NOT Per Capita
	# energy = 'IEA_Residential_Energy_Consumption_Billion_BTU'
	# water = 'USGS_Public_Supply_Total_Water_Withdrawals_MGD'

	df = pandas.read_csv(csvfile, usecols=[statenames,usgs_pop,energy,water,temp,precip])

	x = df[water].values
	x = iea_over_usgs = (df[water]/df[energy])/df[usgs_pop]*1e6
	y = df[precip].values
	fname = False# 'usgs_pubsup_vs_avg_precip.png'
	title = 'Public Supply Water Withdrawals (GPD)\nvs. Average Precipitation (Inches)'
	a = df[statenames].values
	xtitle = energy.replace('_',' ')
	ytitle = temp.replace('_',' ')

	make_plot(x=x,y=y,c='b',s=400,fname=fname,
		title=title,annotations=a,xtitle=xtitle, ytitle=ytitle,
		xlog=False,ylog=False)