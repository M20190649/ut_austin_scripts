import cPickle
import gzip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import scipy
import scipy.linalg
import sgd

class MNISTDataSet: 
	"""A class for reading and plotting the MNIST dataset."""

	def __init__(self,dstype='train',dsrange=None):
		"""Reads in the data file, and if dsrange is specified as a
		slice object, drops everything from the training set other
		than that slice."""
		self.ds = cPickle.load( gzip.open('mnist.pkl.gz') )
		if dsrange:
			self.ds = self.getSlice(dstype,dsrange)
		print self.ds

	def plotDigit(self,adigit,nvals=False): 
		"""Plots an image specified by adigit. If nvals is true, 
		then it uses a colormap to distinguish negative and 
		positive values, as opposed to plotting a monochrome image."""
		imgcmap = cm.gray_r
		imgcmap.set_bad(alpha=0)
		for x in range(len(self.ds)):
			val = self.ds[x][1]
			if val == adigit:
				img = self.ds[x][0]
				dims = int(math.sqrt(len(img)))
				img = img.reshape(dims,dims)
				img = scipy.ma.masked_where(img == 0, img)
				plt.imshow(img,cmap=imgcmap)
		plt.show()

	def plotIdx(self, idx):
		"""Plots the digit at index idx of the training set."""
		img = self.ds[idx][0]
		dims = int(math.sqrt(len(img)))
		img = img.reshape(dims,dims)
		plt.imshow(img, cmap='Greys')
		plt.show()

	def getSlice(self, dsname, aslice=None): 
		"""dsname is one of 'train', 'test', or 'validate' and 
		aslice is a slice. Returns that slice of the datset as
		a tuple."""
		if dsname == 'train': dstemp = self.ds[0]
		elif dsname == 'test': dstemp = self.ds[1]
		elif dsname == 'validate': dstemp = self.ds[2]
		else: 
			dstemp = self.ds[0]
			print 'Invalid dsname entered. Defaulted to training dataset.'
		if aslice == None:
			return dstemp
		else: 
			images = dstemp[0][ aslice[0]:aslice[1] ]
			labels = dstemp[1][ aslice[0]:aslice[1] ]
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

	def bound(self,vec,unitlen=1):
		norm = scipy.sqrt( (vec*vec).sum(axis=1) )
		if norm.ndim == 1: 
			norm = norm.ravel()
		def normalize(vec):
			return vec / norm
		ans = scipy.apply_along_axis(normalize,axis=0,arr=vec)
		norm = norm.reshape(norm.size,1)
		outliers = scipy.where(norm > unitlen)
		ans[outliers] = vec[outliers] / norm[outliers] * unitlen
		ans[norm.ravel() == 0] = 0
		return ans

	def errorInd(self,x,data=None):
		"""Return the number of missclassifications on data, if the parameters
		are specified by x. The array x gives the parameters for all categories.
		If data is none, count the number of missclassifications on the
		validation data set."""
		pass

	def feval(self,x,avg=True):
		"""Evaluate the loss function on the parameters x over the entire
		training data set. If avg is True, average the loss function by the
		number of examples in the training data set."""
		ans = self.funcEval(x,self.ds)
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
		ans = funcEval(x,data)
		if avg:
			ans = ans / (data[1].shape[0]) # average over size of data slice
		return ans

	def grad(self,x,avg=True,bound=True):
		"""Return the gradient of the loss function at the parameters x, evaluated
		over the entire training data set."""
		ans = gradEval(x,self.ds)
		if avg:
			ans = ans / (self.ds[1].shape[0])
		if bound:
			ans = self.bound(ans)
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
		ans = self.gradEval(x,data)
		if avg:
			ans = ans / (data[1].shape[0]) # average over size of data slice
		if bound:
			ans = self.bound(ans)
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
		pass

	def funcEval(self,x,data):
		return scipy.sum( scipy.square( scipy.absolute( x[ data[1] ] - data[0] ) ) )

	def gradEval(self,x,data):
		return ( 2 * scipy.absolute( x[ data[1] ] - data[0] ) )

class MNISTMultiNom(MNISTClassifierBase):

	def classify(self,x,data):
		pass

	def funcEval(self,x,data):
		sum1 = -scipy.sum( ( x[ data[1] ] ).dot( data[0].T ) )
		sum2 = scipy.sum( scipy.log( scipy.sum( scipy.exp( ( x.dot( data[0].T ) ) ), axis=0 ) ) )
		return (sum1 + sum2)

	def gradEval(self,x,data):
		pass

if __name__ == '__main__': 
# 	mnist = MNISTDataSet('train',[0,10])
# 	# mnist.plotIdx(3)
# 	# mnist.plotIdx(6)
# 	# mnist.plotIdx(8)

# 	# mnist.plotDigit(1) # NEEDS WORK. Does not average values

	classifier = MNISTSqLoss()
	x = scipy.zeros((10,784)) 
	sfunc = sgd.step_size(0.9,1)

	sgd = sgd.SGD(afunc=classifier,x0=x,sfunc=sfunc)

	print sgd.getSoln()

# 	for i in range(200):
# 		sgd.nsteps(1)
# 		fname = 'vid5/sgd_q1_{0:03d}'.format(i)
# 		# sgd.plot(alphaMult=0.9)