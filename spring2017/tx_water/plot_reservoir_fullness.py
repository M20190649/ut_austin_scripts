import pandas
import matplotlib.pyplot as plt
import scipy

df = pandas.read_csv('reservoir_annual_pfull_summary.csv')

regions = [chr(i) for i in range(ord('a'),ord('p')+1)]

regions.remove('e')

for l in regions:
	l = l.capitalize()
	print l
	cur_df = df[df['Region'] == l]
	del cur_df['Region']
	countnan = cur_df.isnull().values.sum()
	x = scipy.delete(cur_df.values[0],0)
	y = scipy.array([float(e) for e in cur_df.columns.values])
	if countnan != 0: 
		x_cur = x[countnan-1:]
		y_cur = y[countnan:]
	else: 
		x_cur = x[countnan:]
		y_cur = y[countnan+1:]
	# print x_cur
	print len(y_cur)
	print len(x_cur)

	plt.plot(y_cur,x_cur)
# plt.savefig('reservoir_fullness.png')
plt.show()