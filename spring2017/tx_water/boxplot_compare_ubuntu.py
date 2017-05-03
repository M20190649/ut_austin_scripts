import pandas
import matplotlib.pyplot as plt
import os.path
import scipy
import matplotlib

csvtemp = 'region_{0}.csv'

letters = [chr(i) for i in range(ord('A'),ord('P')+1)]

# x = scipy.array([1000,2000])
# data1 = scipy.array([scipy.random.normal(loc=0.5,size=100),scipy.random.normal(loc=1.5,size=100)]).T
# data2 = scipy.array([scipy.random.normal(loc=2.5,size=100),scipy.random.normal(loc=0.75,size=100)]).T

# print data1
# # print data2

# plt.figure()
# plt.boxplot(data1,0,'',positions=x-100,widths=150)
# # plt.boxplot(data2,0,'',positions=x+100,widths=150)
# plt.xlim(500,2500)
# plt.xticks(x)
# plt.show()

data = []

labels = []

fig,ax = plt.subplots(figsize=(50,30),dpi=100)

matplotlib.rcParams.update({'font.size': 20}) # Change font size everywhere, mainly for point labels

for letter in letters:
	csvfile = csvtemp.format(letter)
	if os.path.isfile(csvfile) and letter != 'P':
		print csvfile
		df = pandas.read_csv(csvfile)#,usecols=['CapitalCost_dollars','Implementation_Status'])
		df = df[df['CapitalCost_dollars'] != 0]
		gb = df.groupby('Summary')
		cur_data = [gb.get_group(x)['CapitalCost_dollars'].values for x in gb.groups]
		data.append((cur_data[0]/1e6))
		data.append((cur_data[1]/1e6))
		labels.append(letter)

for i in range(len(data)):
	boxprops = dict(linestyle='-',linewidth=2,color='peru')
	medianprops = dict(linestyle='-',linewidth=3,color='darkgreen')
	bp = ax.boxplot(data[i],
		positions=[i],
		boxprops=boxprops,
		showfliers=False,
		widths=0.8,
		patch_artist=True,
		# showmeans=True,
		meanline=True,
		medianprops=medianprops)
	for whisker in bp['whiskers']: 
		whisker.set(color='peru',linestyle='-',lw=2)
	for cap in bp['caps']: 
		cap.set(color='peru',linestyle='-',lw=2)
	if i % 2 == 0: # incomplete projects
		for patch in bp['boxes']:
			patch.set_facecolor('tan')
	if i % 2 == 1: # finished projects
		for patch in bp['boxes']:
			patch.set_facecolor('lightblue')

ax.set_xlim(-0.5,len(data)-0.5)

plt.xlabel('Region and Implementation Status',size=40)
plt.ylabel('Capital Cost ($M)',size=40)

plt.xticks(scipy.arange(0.5,len(data)+0.5,2),labels,fontsize=32)

plt.yticks(fontsize=32)

plt.title(
	'Implementation Status vs. Capital Cost of Texas Water Management Strategies',
	size=60,y=1.01)
# plt.grid(True)
plt.plot(-100,0,c='lightblue',label='Complete')
plt.plot(-100,0,c='tan',label='Incomplete')
plt.legend()

# plt.show()
plt.savefig('implementation_vs_capital_costs_all_regions_summary.png')
# plt.boxplot()