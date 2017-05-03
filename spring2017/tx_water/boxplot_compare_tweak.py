import pandas
import matplotlib.pyplot as plt
import os.path
import scipy
import matplotlib

csvtemp = 'region_{0}.csv'

letters = [chr(i) for i in range(ord('a'),ord('p')+1)]

fig,ax = plt.subplots()

matplotlib.rcParams.update({'font.size': 20}) # Change font size everywhere, mainly for point labels

not_implemented = []

producing_water = []

for letter in letters:
	csvfile = csvtemp.format(letter)
	if os.path.isfile(csvfile) and letter != 'p':
		print csvfile
		df = pandas.read_csv(csvfile)#,usecols=['CapitalCost_dollars','Implementation_Status'])
		df = df[df['CapitalCost_dollars'] != 0]
		gb = df.groupby('Summary')
		cur_data = [gb.get_group(x)['CapitalCost_dollars'].values for x in gb.groups]
		# scipy.concatenate( (not_implemented, cur_data[0]))
		not_implemented = not_implemented + list(cur_data[0]/1e6)
		producing_water = producing_water + list(cur_data[1]/1e6)

boxprops1 = dict(linestyle='-',linewidth=2,color='peru')
boxprops2 = dict(linestyle='-',linewidth=2,color='blue')

medianprops = dict(linestyle='-',linewidth=3,color='darkgreen')

bp1 = ax.boxplot(not_implemented,
	positions=[1],
	boxprops=boxprops1,
	showfliers=False,
	widths=0.8,
	patch_artist=True,
	# showmeans=True,
	meanline=True,
	medianprops=medianprops
	)

for whisker in bp1['whiskers']: 
	whisker.set(color='peru',linestyle='-',lw=2)
for cap in bp1['caps']: 
	cap.set(color='peru',linestyle='-',lw=2)
for patch in bp1['boxes']:
	patch.set_facecolor('tan')

bp2 = ax.boxplot(producing_water,
	positions=[2], 
	boxprops=boxprops2,
	showfliers=False,
	widths=0.8,
	patch_artist=True,
	# showmeans=True,
	meanline=True,
	medianprops=medianprops
	)

for whisker in bp2['whiskers']: 
	whisker.set(color='blue',linestyle='-',lw=2)
for cap in bp2['caps']: 
	cap.set(color='blue',linestyle='-',lw=2)
for patch in bp2['boxes']:
	patch.set_facecolor('lightblue')

# for i in range(len(data)):
# 	ax.boxplot(data[i],positions=[i])
	# plt.boxplot(d)
ax.set_xlim(0,3)

plt.xlabel('Region and Implementation Status',size=20)
plt.ylabel('Capital Cost ($M)',size=20)

plt.xticks([1,2],['incomplete','finished'],rotation='60',fontsize=16)

plt.yticks(fontsize=16)

plt.title(
	'Implementation Status vs. Capital Cost\nAverage of All Regions (with data)',
	size=20,y=1.01)

plt.tight_layout()

plt.savefig('implementation_vs_capital_cost_avg_all_regions.png')
# plt.xticks(scipy.arange(1,3),labels=['not_implemented','producing_water'],rotation='vertical')
plt.show()

# plt.boxplot()