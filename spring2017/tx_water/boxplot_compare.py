import pandas
import matplotlib.pyplot as plt
import os.path
import scipy
import matplotlib
import seaborn

csvtemp = 'region_{0}.csv'

letters = [chr(i) for i in range(ord('a'),ord('p')+1)]

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
	if os.path.isfile(csvfile) and letter != 'p':
		print csvfile
		df = pandas.read_csv(csvfile)#,usecols=['CapitalCost_dollars','Implementation_Status'])
		df = df[df['CapitalCost_dollars'] != 0]
		gb = df.groupby('Summary')
		cur_data = [gb.get_group(x)['CapitalCost_dollars'].values for x in gb.groups]
		data.append((cur_data[0]))
		data.append((cur_data[1]))
		labels.append('{0}_in_progress'.format(letter))
		labels.append('{0}_implemented'.format(letter))

print labels

for i in range(len(data)):
	ax.boxplot(data[i],positions=[i])
	# plt.boxplot(d)
ax.set_xlim(-1,len(data))

plt.xlabel('Region and Implementation Status',size=40)
plt.ylabel('Capital Cost ($)',size=40)

plt.xticks(scipy.arange(0,len(data)),labels,rotation='60',fontsize=32)

plt.yticks(fontsize=32)

plt.title(
	'Implementation Status vs. Capital Cost of Texas Water Management Strategies',
	size=60,y=1.01)
plt.grid(True)

plt.show()

# plt.boxplot()