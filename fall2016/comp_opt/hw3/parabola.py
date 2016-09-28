from __future__ import division
import matplotlib.pyplot as plt
import scipy
import random
import scipy.linalg

def bound(vec,unitlen=1):
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

class Parabola: 
	"""Defines a function that looks like:
	Sum_i [ alpha_i ( x_i - c_i )^2 ]
	In other words, a parabola in arbitrary n dimensions."""

	def __init__(self,alpha,center=0):
		"""Initialize.
		alpha == an n-dimensional array defining the alpha_i coefficients
		center == an n-dimensional array defining the c_i constants."""
		self.alpha = alpha
		self.center = scipy.zeros_like(self.alpha)
		self.center += center

	def feval(self,x):
		"""Evaluate the function at x."""
		if x.ndim == 1:
			x = x.reshape(1,x.size)
		return scipy.sum( self.alpha * scipy.power ( (x-self.center) , 2 ) )

	def seval(self,x,ndata=None):
		"""Stochastic evaluation of the function at x. 
		We'll use this in later assignments. For now, this returns the same as feval."""
		return self.feval(x)

	def grad(self,x):
		"""Evaluate the gradient at x."""
		if x.ndim == 1:
			x = x.reshape(1,x.size)
		def gradfx(x):
			return 2 * self.alpha * ( x - self.center ) 
		ans = scipy.apply_along_axis(gradfx,axis=1,arr=x)
		# gradfx = 2 * self.alpha * ( x - self.center )
		return bound(ans)

	def sgrad(self,x,ndata=None):
		"""Return a stochastic gradient at x. 
		Returns the gradient of a uniformly random summand."""
		### Pick random x, return gradient vector for that x
		### in the form [0 0 0 ... x ... 0 0 0 0 0] etc.
		if x.ndim == 1:
			x = x.reshape(1,x.size)
		i = scipy.random.randint(0,x.size)
		def gradfx(x,i):
			return 2 * self.alpha[i] * ( x[i] - self.center[i] ) 
		ans = scipy.apply_along_axis(gradfx,axis=1,arr=x)
		# gradx = 2 * self.alpha[idx] * ( x[0][idx] - self.center[idx] )
		grad = scipy.zeros_like(x)
		ans[0][i] = grad
		return bound(ans)

class ParabolaDir(Parabola):

	def sgrad(self,x,ndata=None):
		"""Returns a stochastic gradient at x.
		Projects the gradient in a uniformly random direction."""
		### Pick a random direction, then calculate (U.T-dot-gradfx)*u
		### to project grad vector in random direction.
		# if ndata == None: ndata = 1
		if x.ndim == 1:
			x = x.reshape(1,x.size)
		grad = self.grad(x)
		u = scipy.randn(*x.shape)
		# u = u / scipy.sqrt( (u*u).sum(axis=1) )
		# u = u / scipy.linalg.norm(u,2,axis=1)
		u = u / scipy.expand_dims(scipy.linalg.norm(u,2,axis=1),1)
		gradfx = u * scipy.sum( (self.grad(x) * u).sum(axis=1) )
		return bound(gradfx)

if __name__ == '__main__':

	alpha = scipy.array([1,2,3,4,5])
	center = scipy.array([6,7,8,9,10])

	test = Parabola(alpha,center)
	test2 = ParabolaDir(alpha,center)


	x = scipy.array([13,2,-2,-3,0])

	# print test.grad(x)

	# print test.sgrad(x)
	# print test2.sgrad(x)
	# plt.scatter(x,test.feval(x))
	# plt.show()