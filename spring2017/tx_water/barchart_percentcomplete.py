import pandas
import matplotlib.pyplot as plt
import scipy

csvfile = 'tx_water_plan_implementation_summary_edit.csv'

df = pandas.read_csv(csvfile)



fig,ax = plt.subplots(1, figsize=(10,5))

cp = df['Complete_Percentage'].values
ip = df['Incomplete_Percentage'].values
ni = df['NoInfo_Percentage'].values

# print cp
# print ip
# print ni



bar_width = 0.7

bar_l = [i for i in range(len(df['Region']))]
print bar_l
1/0

tick_pos = [i+(bar_width/2) for i in bar_l]


ax.bar(bar_l,
	cp,
	label='Complete',
	alpha=0.9,
	color='lightblue',
	width=bar_width,
	edgecolor='white'
	)

ax.bar(bar_l,
	ip,
	bottom=cp,
	label='Incomplete',
	alpha=0.9,
	color='tan',
	width=bar_width,
	edgecolor='white'
	)

ax.bar(bar_l,
	ni,
	bottom=[i+j for i,j in zip(cp,ip)],
	label='No Info',
	alpha=0.9,
	color='grey',
	width=bar_width,
	edgecolor='white'
	)

# ind = scipy.arange(len(df['Region'].values))
# plt.bar(ind,df['Complete_Percentage'].values,color='lightblue')
# plt.xticks(ind + width / 2, df['Region'].values)


plt.xticks(tick_pos,df['Region'])
ax.set_ylabel('Percentage')
# ax.set_xlabel('Region')
plt.xlim([min(tick_pos)-bar_width,max(tick_pos)+bar_width])
plt.ylim(0,100)
# plt.setp(plt.gca().get_xticklabels(),rotation=60,horizontalalignment='right')
plt.title('Percent of Projects Completed')

# Shrink current axis by 10% on bottom
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# Put legend below current axis
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.05),fancybox=True,shadow=True,ncol=5)
# plt.savefig('percent_complete_by_region_barchart.png')
plt.show()