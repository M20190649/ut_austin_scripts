import cPickle
import gzip
import matplotlib.pyplot as plt
import math
import scipy
import sgd

class MNISTDataSet: 
	"""A class for reading and plotting the MNIST dataset."""

	def __init__(self,dstype='train',dsrange=None):
		"""Reads in the data file, and if dsrange is specified as a
		slice object, drops everything from the training set other
		than that slice."""
		self.ds = cPickle.load( gzip.open('mnist.pkl.gz') ) # load dataset
		self.ds = self.getSlice(dstype,dsrange) # if dsrange, take slice of dataset

	def plotDigit(self,adigit,nvals=False): 
		"""Plots an image specified by adigit. If nvals is true, 
		then it uses a colormap to distinguish negative and 
		positive values, as opposed to plotting a monochrome image."""

		# Set max/min of scale equal to +/- of largest value found in adigit
		scale = max( abs( scipy.amax(adigit) ), abs( scipy.amin(adigit) ) )

		# Reshape adigit to square image
		dims = int(math.sqrt(len(adigit)))
		adigit = adigit.reshape(dims,dims)

		# Plot using interpolation
		if nvals: 
			plt.imshow(adigit, cmap='RdBu_r', vmin=-scale, vmax=scale,
				interpolation='nearest') # show image in red-blue colorscale

		# Plot grey if nvals is false
		else: plt.imshow(adigit, cmap='Greys_r') # show image in greyscale

		# Add colorbar and show plot
		plt.colorbar()
		plt.show()

	def plotIdx(self, idx):
		"""Plots the digit at index idx of the training set."""
		img = self.ds[0][idx]
		dims = int(math.sqrt(len(img))) # calculate dimensions of square image
		img = img.reshape(dims,dims) # reshape (ie. from 784 to 28x28)
		plt.imshow(img, cmap='Greys') # show image in greyscale
		plt.show()

	def getSlice(self, dsname, aslice=None): 
		"""dsname is one of 'train', 'test', or 'validate' and 
		aslice is a slice. Returns that slice of the datset as
		a tuple."""
		if dsname == 'train': dstemp = self.ds[0]
		elif dsname == 'test': dstemp = self.ds[1]
		elif dsname == 'validate': dstemp = self.ds[2]
		else: 
			dstemp = self.ds[0] # default to training dataset if none selected
			print 'Invalid dsname entered. Defaulted to training dataset.'
		if aslice == None:
			return dstemp
		else: 
			images = dstemp[0][ aslice[0]:aslice[1] ] # take only images in slice
			labels = dstemp[1][ aslice[0]:aslice[1] ] # take only labels in slice
			return (images,labels)

class MNISTClassifierBase:
	"""Implements the basic operations of classifiers, but is not a classifier. 
	For this class, the parameters are always vectorized. In other words, 
	parameter values x are specified as [x0,x1,x3, \dots] and the functions
	are evaluated at all the parameter values. The only function not vectorized
	is classify."""
	def __init__(self,ds=None):
		"""Read in the dataset, if not passed in. If dataset is passed, assumes
		only one datset (ie. train, test, or validate; not all three)"""
		if not ds:
			ds = cPickle.load( gzip.open('mnist.pkl.gz') )
			self.ds = ds[0] # Use training dataset
		else: 
			self.ds = ds # use passed-in dataset (train, test, or validate)

	def errorInd(self,x,data=None):
		"""Return the number of missclassifications on data, if the parameters
		are specified by x. The array x gives the parameters for all categories.
		If data is none, count the number of missclassifications on the
		validation data set."""
		if not data:
			data = cPickle.load( gzip.open('mnist.pkl.gz') )
			vals = data[2][1] # values of validation dataset
		else: 
			vals = data[1] # values of passed-in dataset
		ans = scipy.sum( x != vals ) # number of mismatches
		print 'Error: {0}/{1}'.format(ans,vals.shape[0])
		return ans

	def feval(self,x,avg=True):
		"""Evaluate the loss function on the parameters x over the entire
		training data set. If avg is True, average the loss function by the
		number of examples in the training data set."""
		ans = self.funcEval(x,self.ds) # evaluage function for all data
		if avg:
			ans = ans / (self.ds[1].shape[0])
		return ans

	def sfeval(self,x,ndata=100,avg=True):
		"""Stochastic evaluation of the loss function on the parameters x over
		the training data set.
		ndata -- how many samples to take from the training data set
		avg -- if True, average the loss function by the number of examples samples"""
		u = scipy.random.randint(0,x.shape[0],ndata) # ndata random index values
		data_vals = scipy.take(self.ds[0],u,axis=0) # take values at random indices
		data_cats = scipy.take(self.ds[1],u,axis=0) # take categories at random indices
		data = (data_vals,data_cats) # put data into correct format
		ans = funcEval(x,data) # evaluate function only for stochastically selected data
		if avg:
			ans = ans / (data[1].shape[0]) # average over size of data slice
		return ans

	def grad(self,x,avg=True,bound=True):
		"""Return the gradient of the loss function at the parameters x, evaluated
		over the entire training data set."""
		ans = gradEval(x,self.ds) # evaluate gradient for all data
		if avg:
			ans = ans / (self.ds[1].shape[0]) # bound data
		if bound:
			ans = sgd.bound(ans)
		return ans

	def sgrad(self,x,ndata=100,bound=True,avg=True):
		"""Return a stochastic gradient at x, evaluated at ndata samples from the
		training set. 
		x -- the parameters at which to evaluate the stochastic gradient
		ndata -- the number of samples from the training set to take
		bound -- whether to bound the gradient
		avg -- whether to average the gradient by the number of samples taken"""
		u = scipy.random.randint(0,x.shape[0],ndata) # ndata random index values
		data_vals = scipy.take(self.ds[0],u,axis=0) # take values at the random indices
		data_cats = scipy.take(self.ds[1],u,axis=0) # take categories at the random indices
		data = (data_vals,data_cats) # put data into correct format
		ans = self.gradEval(x,data) # evaluate gradient only for stochastically selected data
		if avg:
			ans = ans / (data[1].shape[0]) # average over size of data slice
		if bound:
			ans = sgd.bound(ans) # bound data
		return ans

	def classify(self,x,data):
		"""Use the parameters in x to return classes for the examples in data.
		(Not implemented)
		This is the only non-vectorized function in this class. For all other 
		functions, x is a [x0,x1,x2] array. 
		x -- the parameters to use for classification
		data -- a list or array of example images
		retval -- an array of categories, one for each example in data"""
		pass

	def funcEval(self,x,data):
		"""Evaluate the loss function at parameters x, and data. 
		(Not implemented)
		x -- the parameters of the classifier
		data -- (input data, labels)
		output -- function value"""
		pass

	def gradEval(self,x,data):
		"""Evaluate the gradient of the loss function at parameters x and data. 
		(Not implemented)
		x -- the parameters of the classifier
		data -- (input data, labels)
		output -- gradient evaluated at the data"""
		pass

class MNISTSqLoss(MNISTClassifierBase):

	def classify(self,x,data):
		for i in range(x.shape[0]): # iterate over all parameters (0 through 9)
			# Calculate residuals for all images in data, using ith parameter
			temp = scipy.sum( scipy.power( scipy.subtract(data,x[i]), 2) , axis=1 )
			# For first iteration, set all answers to zero
			if i == 0:
				retval = scipy.zeros((temp.shape[0]))
				least = scipy.copy(temp)
			# If current residuals are less than previous, 
			else: 
				idx = scipy.where(temp < least)[0] # find where current sqloss is less
				retval[idx] = i
				least[idx] = temp[idx]
		return retval

	def funcEval(self,x,data):
		return scipy.sum( scipy.power( x[ data[1] ] - data[0] ), 2 )

	def gradEval(self,x,data):
		
		sqloss = 2 * ( x[ data[1] ] - data[0] )
		newx = scipy.zeros((x.shape))

		for i in range(x.shape[0]):
			idx = scipy.where(data[1] == i)
			sqlosstemp = sqloss[idx]
			newx[i] += scipy.sum(sqlosstemp, axis=0)

		# def gradarray(iterator):
		# 	param = data[1][iterator]
		# 	datapt = data[0][iterator]
		# 	sqloss = 2 * ( x[ param ] - datapt )
		# 	print scipy.sum(sqloss)
		# 	newx[param] += sqloss
		# 	print 'newx', scipy.sum(newx)
		# 	return iterator
		# newx = x
		# iterator = scipy.arange(data[1].shape[0])
		# iterator = iterator.reshape((1,iterator.shape[0]))
		# scipy.apply_along_axis(gradarray, axis=0, arr=iterator)

		return newx

class MNISTMultiNom(MNISTClassifierBase):

	def classify(self,x,data):
		for i in range(x.shape[0]): 
			temp = - (data).dot(x[i])
			if i == 0:
				retval = scipy.zeros((temp.shape[0]))
				prev = temp
			else: 
				idx = scipy.where(temp < prev)[0]
				retval[idx] = i
				prev = temp
		return scipy.real( retval )

	def funcEval(self,x,data):

		denom = scipy.log( scipy.exp( data[0] * x[ data[1] ] ) )
		ans = scipy.sum( -(data[0] * x[ data[1] ]) + denom , axis=1 )
		return ans

		# sum1 = -scipy.sum( (x[ data[1] ]).dot((data[0]).T) )
		# sum2 = scipy.sum( scipy.log( scipy.sum( scipy.exp( x.dot((data[0]).T) ), axis=0 ) ) )
		# return (sum1 + sum2)

	def gradEval(self,x,data):
		a = data[0]
		w = x[ data[1] ]
		wi = x
		# print w.dot(a.T)

		mnom = -(x[ data[1] ]).dot(data[0].T)
		# print mnom.shape

		numer = a * scipy.exp( w.dot(a.T) )
		# print numer.shape

		denom = scipy.sum( scipy.exp( wi.dot(a.T) ) )
		temp = -a + numer / denom



		newx = scipy.zeros((sqloss.shape))



		sqloss = 2 * ( x[ data[1] ] - data[0] )
		newx = scipy.zeros((x.shape))

		for i in range(x.shape[0]):
			idx = scipy.where(data[1] == i)
			sqlosstemp = sqloss[idx]
			newx[i] += scipy.sum(sqlosstemp, axis=0)

		# def gradarray(iterator):
		# 	idx = data[1][iterator]
		# 	a = data[0][iterator]
		# 	w = x[ data[1][iterator] ]
		# 	wi = x
		# 	numer = a * scipy.exp( w.dot(a.T) )
		# 	denom = scipy.sum( scipy.exp( wi.dot(a.T) ) )
		# 	temp = -a + numer / denom
		# 	newx[idx] += temp
		# 	return iterator
		# newx = x
		# iterator = scipy.arange(data[1].shape[0])
		# iterator = iterator.reshape((1,iterator.shape[0]))
		# scipy.apply_along_axis(gradarray, axis=0, arr=iterator)
		return newx	

if __name__ == '__main__': 
	train = MNISTDataSet('train')
	test = MNISTDataSet('test')

	train.plotIdx(0) # first figure

	# K-Means
	loss = MNISTSqLoss(train.ds)

	x = scipy.zeros((10,784)) 
	grad = loss.gradEval(x, (train.ds[0][:50], train.ds[1][:50]))
	train.plotDigit(grad[3],nvals=True) # second figure

	sfunc = sgd.step_size(0.9,1)
	opt = sgd.SGD(afunc=loss,x0=x,sfunc=sfunc,histsize=500,ndata=300,keepobj=False)
	opt.nsteps(500)
	ans = opt.getAvgSoln(100)

	result = loss.classify(ans,test.ds[0]) 
	loss.errorInd(result)
	print 'Sum of parameters:', scipy.sum(result)

	# MultiNom
	multinom = MNISTMultiNom(train.ds)

	grad = multinom.gradEval(x, (train.ds[0][:50], train.ds[1][:50]))
	train.plotDigit(grad[3],nvals=True) # third figure

	sfunc = sgd.step_size(0.9,1)
	opt = sgd.SGD(afunc=multinom,x0=x,sfunc=sfunc,histsize=500,ndata=300,keepobj=False)
	opt.nsteps(500)
	ans = opt.getAvgSoln(100)
	
	result = multinom.classify(ans,test.ds[0]) 
	multinom.errorInd(result)
	print 'Sum of parameters:', scipy.sum(result)

	# scipy.set_printoptions(threshold='nan')