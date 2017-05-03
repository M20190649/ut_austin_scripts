# from __future__ import division
import scipy
import parabola

def step_size(gamma,start=1):
	"""Return a function that produces a step size for stochastic
	gradient descent. The step size for the nth step is of the form:
	start / n^gamma
	For stochastic gradient descent to work, gamma has to be less
	than or equal to 1. The parameter start defines the length of the 
	first step. 
	Returns a function sfunc such that sfunc(k) returns the length of
	the kth step."""
	if gamma > 1 or gamma <= 0:
		raise ValueError('Gamma must be within (0,1]')
	def sfunc(k):
		return start*1.0/scipy.power(k,gamma)
	return sfunc

class SGD:
	"""A class implementing stochastic gradient descent."""

	def __init__(self,afunc,x0,sfunc,proj=None,histsize=-1,smallhist=False,ndata=100,keepobj=True):
		"""afunc -- the objective function, containing... 
		afunc.sgrad(x,ndata) returning a stochastic subgradient,
		afunc.afunc.feval(x) returning a function evaluation, and
		afunc.sfeval(x,ndata) returning a stochastic function evaluation.
		x0 -- initial point.
		sfunc == a step function. sfunc(n) returns the size of the nth step.
		proj -- a projection function. proj(x) returns the closest point to x within the feasible region.
		histsize -- how many steps of history to maintain (-1 is all the steps).
		smallhist -- whether to maintain the history of gradients, stepsizes, and pre-projection points.
		ndata -- the number of data points to pass into sgrad and sfeval.
		keepobj -- whether to maintain a history of the objective function value."""
		self.afunc = afunc
		self.sfunc = sfunc
		self.proj = proj
		if self.proj == None:
			def ident(x):
				return x
			self.proj = ident
		self.histsize = histsize
		self.smallhist = smallhist
		self.ndata = ndata
		self.keepobj = keepobj
		self.setStart(x0)
		self.reset()

	def setStart(self,x0):
		"""Set the start point."""
		self.x0 = x0

	def reset(self):
		"""Reset the history of the optimization. In other words, 
		drop all history and start again from x0 and 1st step."""
		self.stepcount = 1
		self.g_hist = []
		self.s_hist = []
		self.sx_hist = []
		self.x_hist = []
		self.obj_hist = []
		self.setStart(self.x0)
		self.x_hist.append(self.x0)

	def dostep(self):
		"""Take a single step of SGD."""
		x = self.x_hist[-1]
		grad = self.afunc.sgrad( x, self.ndata )
		step = self.sfunc( self.stepcount )
		nx = x - step * grad
		nxproj = self.proj(nx)
		self.stepcount += 1
		self.x_hist.append(nxproj)
		if not self.smallhist:
			self.g_hist.append(grad)
			self.s_hist.append(step)
			self.sx_hist.append(nx)
		# if self.keepobj:
		# 	self.obj_hist.append( self.afunc.sfeval( nxproj, self.ndata ) )
		if self.histsize > 0 and len(self.x_hist) > self.histsize:
			del self.x_hist[0]
			if not self.smallhist:
				del self.g_hist[0]
				del self.s_hist[0]
				del self.sx_hist[0]
			if self.keepobj:
				del self.obj_hist[0]

	def nsteps(self,an=1):
		"""Take an steps of SGD."""
		for _ in range(an):
			self.dostep()

	def getAvgSoln(self,wsize=10):
		"""Average the last wsize points and return a solution."""
		return scipy.average( self.x_hist[-wsize:], axis=0 )

	def getSoln(self,wsize=10,winterval=1,abstol=1e-6,reltol=1e-6):
		"""Keep performing SGD steps until: afunc.feval(x*_prev) and 
		afunc.feval(x*) are within the specified tolerances.
		x* -- is a solution obtained from averaging the last wsize points.
		x*_prev -- is a solution obtained by averaging the wsize points that
		were wsize*(winterval_1) back in history.
		Intuitively, this function keeps performing steps until "the objective 
		value" doesn't change much. Be careful because it involves calls to 
		afunc.feval	that may be slow."""
		nreq = wsize*(winterval+2)
		fold = abstol
		fnew = -abstol

		abstest = 0
		reltestmore = 0
		reltestless = 0

		def step():
			self.nsteps( nreq )
			xold = self.getAvgSoln(wsize)
			xnew = scipy.average( self.x_hist[ -nreq:(-nreq+wsize) ], axis=0 )
			fold = self.afunc.feval(xold)
			fnew = self.afunc.feval(xnew)
			return (fold,fnew)

		while abstest != 1:
			fold,fnew = step()
			if scipy.all( scipy.absolute(fold - fnew) < abstol):
				abstest += 1
		while reltestmore != 1:
			fold,fnew = step()
			if scipy.all( float(fold+1e-11)/(fnew-1e-11) < 1 + reltol):
				reltestmore += 1
		while reltestless != 1:
			fold,fnew = step()
			if scipy.all( float(fold+1e-11)/(fnew-1e-11) < 1 - reltol):
				reltestless += 1

		return self.stepcount

	def plot(self,fname=None,n=100,alphaMult=1,axis=None):
		"""Produce a plot of the last n SGD steps. 
		fname -- a file name where to save the plot, or show if None.
		n -- the number of points to display.
		alphaMult -- a geometric multiplier on the alpha value of the segments, 
		with the most recent one having alpha=1.
		axis -- the axis on which to plot the steps."""
		import matplotlib.pyplot as plt

		n = min(n,len(self.x_hist))
		points = scipy.array(self.x_hist[-n:])

		if not axis:
			ax = plt.gca()
		else:
			ax = axis

		plt.clf()
		fig = plt.figure()

		colors = ['black','g','b','r']
		alpha = 1

		if points.shape[2] == 3:
			from mpl_toolkits.mplot3d import Axes3D	
			import mpl_toolkits.mplot3d
			ax = fig.gca(projection='3d')
			ax.scatter(*zip(self.afunc.center),color='red',marker='o',s=40)

		if points.shape[2] == 2:
			plt.scatter(*zip(self.afunc.center),color='red',marker='o',s=40)

		for i in range(points.shape[1]):
			print points[:,i,:]
			pts = points[:,i,:]
			plt.plot(*zip(*pts), color=colors[i%len(colors)], alpha=alpha)	

		if fname == None:
			plt.show()
		else:
			fig.savefig(fname)
		plt.close()