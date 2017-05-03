import pandas
import scipy

df = pandas.read_csv('hw4_KTtest.csv') # csv containing data
x = df['Median grain size (x)'].values # x-vals (precipitation)
y = df['Yield (y)'].values # y-vals (years)
x = scipy.log(x) # transform to ln(x)
y = scipy.log(y) # transform to ln(y)

n = len(x) # number of items in dataset

ivals = scipy.arange(n-1) # i from (0,1,2,...,n-2)
jvals = scipy.arange(1,n) # j-vals (1,2,3,...,n-1)

vals = []
for i in ivals: 
	for j in jvals:
		if i < j:
			val = ( y[j] - y[i] ) / ( x[j] - x[i] ) # calculate slope
			vals.append( val ) # add slopes to list

b1 = scipy.median( vals ) # compute median of slopes in list
b0 = scipy.median(y) - b1 * scipy.median(x) # ymed - b1 * xmed
print 'b1: %.3f' % b1
print 'b0: %.3f' % b0
print 'equation: y = %.3f + %.3fx' % (b0,b1)

