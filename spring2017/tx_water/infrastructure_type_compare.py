import pandas
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import Counter

csvtemp = 'region_{0}.csv'

letters = [chr(i) for i in range(ord('a'),ord('p')+1)]

fig,ax = plt.subplots()

matplotlib.rcParams.update({'font.size': 20}) # Change font size everywhere, mainly for point labels

producing_water = []
not_implemented = []

for letter in letters:
	csvfile = csvtemp.format(letter)
	if os.path.isfile(csvfile) and letter != 'p':
		print csvfile
		df = pandas.read_csv(csvfile,usecols=['Infrastructure_Type','Summary','Implementation_Status'])
		df = df.dropna()
		providing = df[df['Implementation_Status'] == 'Providing Water']
		failing = df[df['Implementation_Status'] == 'Not Functional']
		
		producing_water = producing_water + list(providing.Infrastructure_Type.values)
		not_implemented = not_implemented + list(failing.Infrastructure_Type.values)
		print Counter(providing.Infrastructure_Type.values)
		print Counter(failing.Infrastructure_Type.values)
# print producing_water

water_dict = Counter(producing_water)
nowater_dict = Counter(not_implemented)

pw = []
pn = []
count = []

for key in nowater_dict.keys():
	w = water_dict[key]
	n = nowater_dict[key]
	s = float(w + n)
	pw.append( w / s * 100 )
	pn.append( n / s * 100 )
	count.append( w + n )
	# print perc_w, perc_n, (perc_w + perc_n)

bar_width = 0.7

bar_l = [i for i in range(len(nowater_dict))]

tick_pos = [i+(bar_width/2) for i in bar_l]

ax.bar(bar_l,
	pw,
	label='Complete',
	alpha=0.9,
	color='lightblue',
	width=bar_width,
	edgecolor='white'
	)

ax.bar(bar_l,
	pn,
	bottom=pw,
	label='Incomplete',
	alpha=0.9,
	color='tan',
	width=bar_width,
	edgecolor='white'
	)

labels = ['{0} ({1})'.format(str(i),j) for i,j in zip(nowater_dict.keys(),count)]

plt.xticks(tick_pos,labels,rotation='45',ha='right')
ax.set_ylabel('Percentage')
# ax.set_xlabel('Region')
plt.xlim([min(tick_pos)-bar_width,max(tick_pos)+bar_width])
plt.ylim(0,100)
# plt.setp(plt.gca().get_xticklabels(),rotation=60,horizontalalignment='right')
plt.title('Percent of Projects Completed\nby Infrastructure Type')

# Shrink current axis by 20% on right
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

# Put legend below current axis
ax.legend(loc='lower center', bbox_to_anchor=(.8, -2),fancybox=True,shadow=True)

plt.tight_layout()

# plt.savefig('percent_complete_by_infrastructure_type_barchart.png')
# plt.show()