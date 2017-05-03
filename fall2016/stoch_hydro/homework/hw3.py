import pandas
import scipy
import scipy.stats
import matplotlib.pyplot as plt

class Hw3part1:

	def __init__(self,dfloc,k,l,mu,stdev):
		self.df = pandas.read_csv(dfloc,usecols=['Year','Precipitation[in]']) # read in data
		self.k = k
		self.l = l
		self.theta = 1/l # scale = 1/rate
		self.mu = mu
		self.stdev = stdev
		self.var = self.stdev**2

	def plot_gamma(self, filename=None):
		xlim = scipy.stats.gamma.ppf(0.9999,a=self.k,scale=self.theta)
		x_axis = scipy.linspace(0, xlim, 30) # xnum steps
		fig, ax = plt.subplots()
		ax.plot(x_axis, scipy.stats.gamma.pdf(x_axis,a=self.k,scale=self.theta))
		plt.grid()
		ax.set_title('2-Parameter Gamma Distribution, G({0},{1})'.format(self.k,self.l))
		if filename: fig.savefig(self.output + filename)
		else: plt.show()

	def plot_norm(self, filename=None):
		xlim = scipy.stats.norm.ppf(0.9999,loc=self.mu,scale=self.var)
		x_axis = scipy.linspace(0, xlim, 30) # xnum steps
		fig, ax = plt.subplots()
		ax.plot(x_axis, scipy.stats.norm.pdf(x_axis,self.mu,self.var))
		plt.grid()
		ax.set_title('Normal Distribution, N({0},{1})'.format(self.mu,self.var))
		if filename: fig.savefig(self.output + filename)
		else: plt.show()

	def get_gamma_cdf(self,x):
		""" Find probability of gamma distribution at value x"""
		ans = scipy.stats.gamma.cdf(x,a=self.k,scale=self.theta)
		# ans = scipy.stats.gamma.ppf(x,a=self.k,scale=self.scale)
		return ans

	def get_norm_cdf(self,x):
		""" Find probability of normal distribution at value x"""
		ans = scipy.stats.norm.cdf(x,loc=self.mu,scale=self.var)
		return ans

	def chisq(self,ranges=[0,1],dist='norm'):
		""" Computes the chisq test statistic and p-value for classes defined by ranges[]
			on a gamma distribution defined by the provided k and lambda values"""
		print '--- Chisq test on gamma distribution, k = {0}, scale = {1} ---'.format(self.k,self.l)
		obs = []
		exp = []
		for i in range(1,len(ranges)):
			data = self.df[self.df['Precipitation[in]'].values < ranges[i]]['Precipitation[in]']
			data = data[data >= ranges[i-1]].values
			iobs = len(data)*1.0/len(self.df)
			if dist == 'norm': iexp = self.get_norm_cdf(ranges[i])
			elif dist == 'gamma': iexp = self.get_gamma_cdf(ranges[i])
			obs.append(iobs)
			exp.append(iexp)
		chisq,p = scipy.stats.chisquare(f_obs=obs,f_exp=exp,ddof=0)
		print 'Observations: ', obs
		print 'Expectations: ', exp
		print 'Test Statistic: ', chisq
		print 'P-Value: ', p
		if (p>=0.95): print 'Accept null hypothesis'
		else: print 'Reject null hypothesis'
		print '\n'
		return chisq,p

	def ks(self,dist='norm'):
		""" Computes the Kolmogorov-Smirnov test statistic and p-value
			on a gamma distribution defined by the provided k and lambda values"""
		print '--- Kolmogorov-Smirnov test on gamma distribution, k = {0}, scale = {1} ---'.format(self.k,self.l)
		data = self.df['Precipitation[in]'].values
		ranks = scipy.stats.rankdata(data,method='average')
		empcdf = ranks*1.0/(len(data)+1)
		theocdf = self.get_gamma_cdf(data)
		if dist == 'norm': 
			ksD,p = scipy.stats.kstest(data,lambda x: self.get_norm_cdf(x),N=len(data),mode='approx')
		elif dist == 'gamma': 
			ksD,p = scipy.stats.kstest(data,lambda x: self.get_gamma_cdf(x),N=len(data),mode='approx')
		# ksD,p = scipy.stats.kstest(data,'gamma',args=(self.k,0,self.theta),N=len(data),mode='approx')
		# ksD,p = scipy.stats.kstest(data,'norm',args=(self.mu,self.var),N=len(data),mode='approx')
		print 'Empirical CDF: ', empcdf
		print 'Theoretical CDF: ', theocdf
		print 'Test Statistic: ', ksD
		print 'P-Value: ', p
		if (p>=0.95): print 'Accept null hypothesis'
		else: print 'Reject null hypothesis'
		print '\n'
		return ksD,p

class Hw3part2():

	def __init__(self,dfloc):
		self.df = pandas.read_csv(dfloc) # read in data
		self.data = self.df['peak_streamflow_cfs'].values
		self.mu = scipy.average(self.data)
		self.var = scipy.var(self.data)

	def plot_norm(self, filename=None):
		xlim = scipy.stats.norm.ppf(0.9999,loc=self.mu,scale=self.var)
		x_axis = scipy.linspace(-xlim, xlim, 100) # xnum steps
		fig, ax = plt.subplots()
		# ax.hist(self.data)
		ax.plot(x_axis, scipy.stats.norm.pdf(x_axis,self.mu,self.var))
		plt.grid()
		ax.set_title('Normal Distribution, N({0},{1})'.format(self.mu,self.var))
		if filename: fig.savefig(self.output + filename)
		else: plt.show()

if __name__ == '__main__':

	# Part 1
	# dfpart1 = 'hw3given1.csv'
	# k = 3.76
	# l = 1.923 # rate, in^-1
	# mu = 1.96 # in
	# stdev = 1.12 # in
	# hwpart1 = Hw3part1(dfpart1,k,l,mu,stdev)
	# hwpart1.plot_norm()
	# hwpart1.chisq(ranges=[0,1,1.5,2,2.5,3,float("inf")],dist='gamma')
	# hwpart1.chisq(ranges=[0,1,1.5,2,2.5,3,float("inf")],dist='norm')
	# hwpart1.ks(dist='gamma')
	# hwpart1.ks(dist='norm')

	# Part 2
	dfpart2 = 'hw1given.csv'
	hwpart2 = Hw3part2(dfpart2)
	hwpart2.plot_norm()