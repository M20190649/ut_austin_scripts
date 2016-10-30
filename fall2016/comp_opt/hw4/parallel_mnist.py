import ipyparallel
import mnist
import engine_data_store
import scipy
import timeit

def engine_object_setup(arg):
	"""Set up the data in the engine_data_store module inside each engine."""
	(i,total) = arg
	import ipyparallel
	import mnist
	import engine_data_store
	import scipy
	datasize = 50000
	data = datasize/total
	if i < (total-1):
		engine_data_store.ds = mnist.MNISTDataSet(dsrange=slice(data*(i),data*(i+1)))
	else: 
		engine_data_store.ds = mnist.MNISTDataSet(dsrange=slice(data*(i),datasize))
	engine_data_store.loss = mnist.MNISTSqLoss(ds=engine_data_store.ds)
	return 1

def apply_loss_func(fname,*args,**kwargs):
	"""Apply a loss object's function inside an engine, fname is the 
	name of the function to call."""
	if fname == 'errorInd': engine_data_store.loss.errorInd(*args,**kwargs)
	if fname == 'sgrad': engine_data_store.loss.sgrad(*args,**kwargs)
	if fname == 'sfeval': engine_data_store.loss.sfeval(*args,**kwargs)
	if fname == 'feval': engine_data_store.loss.feval(*args,**kwargs)
	if fname == 'classify': engine_data_store.loss.classify(*args,**kwargs)
	if fname == 'funcEval': engine_data_store.loss.funcEval(*args,**kwargs)
	if fname == 'gradEval': engine_data_store.loss.gradEval(*args,**kwargs)

def split_n_calls(n,k):
	"""Split a given number n into k uniformly random, positive summands."""
	ans = scipy.zeros(k)
	ans[:-1] = scipy.random.randint(0,n,size=(k-1))
	ans[-1] = n
	ans.sort()
	ans[1:] = ans[1:] - ans[:-1]
	return ans

class ParallelLoss:
	"""Implement a loss in a parallel way, by splitting the dataset across
	many engines."""

	def __init__(self,client):
		"""Save the ipyparallel client."""
		self.c = client
		self.dv = c.direct_view()
		self.numeng = len(client.ids)

	def classify(self,x,data):
		"""Just call classify on engine 0, because this operation requires
		none of the dataset."""
		ans = self.client[0].apply(apply_loss_func,'classify',x,data).get()
		self.client.results.clear()
		return ans

	def errorInd(self,x,data=None):
		"""Return the number of misclassifications on data, with the parameters
		specified by x. The array x gives the parameters for all categories.

		If data is none, count the number of misclassifications on the
		validation dataset."""
		if not data: data = self.ds.validation
		ans = self.classify(x,data[0])
		return scipy.sum( data[1] != ans)

	def sgrad(self,x,ndata):
		"""Split the ndata samples across the engines uniformly, call sgrad
		at each engine, and combine the results."""
		res = []
		ans = 0
		split = split_n_calls(ndata,self.numeng)
		for eng, data in zip(self.client, split):
			if ndata > 0: 
				tempres = eng.apply( apply_loss_func,'sgrad',x,data,bound=False,avg=False )
				res.append( tempres )
		for r in res:
			ans += r.get()
		self.client.results.clear()
		return mnist.bound(ans/data)

	def sfeval(self,x,ndata):
		"""Split the ndata samples across the engines uniformly, call sfeval
		at each engine, and combine the results."""
		res = []
		ans = 0
		split = split_n_calls(ndata,self.numeng)
		for eng, data in zip(self.client, split):
			if ndata > 0: 
				tempres = eng.apply( apply_loss_func,'sfeval',x,data,avg=False )
				res.append( tempres )
		for r in res:
			ans += r.get()
		self.client.results.clear()
		return ans/data

	def feval(self,x,ndata):
		"""Call feval at each engine and combine the results."""
		res = []
		ans = 0
		split = split_n_calls(ndata,self.numeng)
		for eng, data in zip(self.client, split):
			tempres = eng.apply( apply_loss_func,'feval',x,avg=False )
			res.append( tempres )
		for r in res:
			ans += r.get()
		self.client.results.clear()
		return ans / 50000

if __name__ == '__main__':
	import ipyparallel
	client = ipyparallel.Client()
	numeng = len(client.ids)
	startup = [(i, numeng) for i in client.ids]
	res = []
	ans = []
	for eng, arg in zip(client, startup):
		res.append( eng.apply(engine_object_setup, arg) )
	for r in res:
		ans.append(r.get()) # Not sure why this isn't working; says slice can't 'getitem'
	print ans