import statswrap
import pylab
import matplotlib.pyplot as plt
import scipy

yeardf = statswrap.Statistics('hw2given_1.csv', 'Year', 'prob1/')
xdf = statswrap.Statistics('hw2given_1.csv', 'x', 'prob1/')
ydf = statswrap.Statistics('hw2given_1.csv', 'y', 'prob1/')

bp1df = statswrap.Statistics('hw2given_3.csv', 'BP-1', 'prob3/')
bp2df = statswrap.Statistics('hw2given_3.csv', 'BP-2', 'prob3/')
bp3df = statswrap.Statistics('hw2given_3.csv', 'BP-3', 'prob3/')

# print xdf

x = xdf.values
y = ydf.values
year = yeardf.values

bp1 = bp1df.values
bp2 = bp2df.values
bp3 = bp3df.values

# def multi_boxplot(*args):
# 	fig, ax = plt.subplots()
# 	fig.set_size_inches(20,16, forward=True)
# 	ax.boxplot(args, 0, sym='rs', vert=0)
# 	ax.set_title('Boxplot of pH samples for BP-1, BP-2, & BP-3',fontsize=42)
# 	ax.set_yticklabels(['BP-1','BP-2','BP-3'])
# 	ax.tick_params(labelsize=28)
# 	# ax.grid(linestyle='-',color='lightgrey')
# 	# ytickNames = plt.setp(ax,yticklabels=['BP-1','BP-2','BP-3'])
# 	# plt.setp(ytickNames,fontsize=20)
# 	# plt.setp(xtickNames,fontsize=20)
# 	fig.savefig('prob3/multi_boxplot.pdf')
# 	# plt.legend(loc='upper left',fontsize=24)
# 	plt.show()

def multi_boxplot(*args):
	fig, ax = plt.subplots()
	fig.set_size_inches(20,16, forward=True)
	# ax.grid(linestyle='-',color='lightgrey')
	ax.boxplot(args, 0, sym='rs', vert=0)
	ax.set_title('Boxplot of thunderstorms reported for 21 years',fontsize=42)
	ax.set_yticklabels(['Northeastern United States','Great Lakes states'],rotation=90)
	ax.tick_params(labelsize=28)
	# ytickNames = plt.setp(ax,yticklabels=['BP-1','BP-2','BP-3'])
	# plt.setp(ytickNames,fontsize=20)
	# plt.setp(xtickNames,fontsize=20)
	fig.savefig('prob1/multi_boxplot.pdf')
	plt.show()

def scatter(x,y):
	fig, ax = plt.subplots()
	fig.set_size_inches(20,16, forward=True)
	ax.scatter(x,y,s=140,c='b')
	ax.set_xlabel('Northeastern US - Thunderstorms per year',fontsize=28)
	ax.set_ylabel('Great Lakes states - Thunderstorms per year',fontsize=28)
	# ax.scatter(year,y,s=140,c='r',label='Great Lakes states')
	ax.set_title('Scatterplot of thunderstorms reported for 21 years',fontsize=42)
	ax.tick_params(labelsize=28)
	# plt.legend(loc='upper left',fontsize=24)
	fig.savefig('prob1/scatter.pdf')
	plt.show()

# multi_boxplot(x,y)
# multi_boxplot(x,y)
scatter(x,y)

# multi_boxplot()