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
		# i = 0
		# while i < an:
		# 	self.dostep()
		# 	i += 1

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
		while (scipy.all( scipy.absolute(fold - fnew) >= abstol)) or \
			(scipy.all( float(fold+1e-11)/(fnew-1e-11) >= 1 + reltol)) or \
			(scipy.all( float(fold+1e-11)/(fnew-1e-11) >= 1 - reltol)):
			self.nsteps( nreq )
			xold = self.getAvgSoln(wsize)
			xnew = scipy.average( self.x_hist[ -nreq:(-nreq+wsize) ], axis=0 )
			fold = self.afunc.feval(xold)
			fnew = self.afunc.feval(xnew)
			# print scipy.absolute(fold-fnew), (fold+1e-11)/(fnew-1e-11)
		return self.stepcount

	def plot(self,fname=None,n=100,alphaMult=1,axis=None):
		"""Produce a plot of the last n SGD steps. 
		fname -- a file name where to save the plot, or show if None.
		n -- the number of points to display.
		alphaMult -- a geometric multiplier on the alpha value of the segments, 
		with the most recent one having alpha=1.
		axis -- the axis on which to plot the steps."""
		import matplotlib.pyplot as plt
		from mpl_toolkits.mplot3d import Axes3D		
		pts = scipy.array(self.x_hist)
		n = min(n,pts.shape[0])
		pts = scipy.array(self.x_hist[-n:])
		plt.clf()
		fig = plt.figure()
		alpha = 1
		
		if not axis:
			ax = plt.gca()
		else:
			ax = axis
		x = []
		y = []
		z = []
		a = []
		pairs = []
		colors = ['black','g','b','r']
		if self.x0.ndim == 1:
			self.x0 = self.x0.reshape(1,self.x0.size)
		if self.x0.shape[1] == 2:
			for k in range(self.x0.shape[0]): # num rows
				for i in range(len(pts)-1):
					xnew = pts[i][k]
					if xnew.size == 1: xnew = pts[-i]
					ynew = pts[i+1][k]
					if ynew.size == 1: ynew = pts[-(i-1)]
					pairs.append( zip(xnew,ynew) ) # next x and y after a step
					plt.cla()
					colors_array = scipy.arange(self.x0.shape[0])
					colors_array = scipy.repeat(colors_array,(len(pts)-1))
					a = 0.0
					for j in range(len(pairs)): # all x,y pairs
						if (j*self.x0.shape[0]) > ( len(pairs) - (15*self.x0.shape[0]) ): 
							a = 1.0
							a = scipy.absolute( 1 - ( len(pairs) - j ) / float(len(pairs)) )
							if j == 0: a = 1.0/(i*2+1)
							if j > 15: a = scipy.absolute( 1 - (len(pairs) - 15 - j) / 15.0 )
						if self.x0.shape[0] > 1: # vectorized x input
							a = 1.0 # temporary over-ride
						ax.plot(*pairs[j][-10:], c=colors[colors_array[j]], alpha=a )
						# a *= alphaMult
				ax.scatter(2,1,color='red',marker='o',s=40)

		if self.x0.shape[1] == 3:
			import mpl_toolkits.mplot3d
			ax = fig.gca(projection='3d')
			for i in range(len(pts)-1):
				xnew = pts[i][0]
				if xnew.size == 1: xnew = pts[-i]
				ynew = pts[i+1][0]
				if ynew.size == 1: ynew = pts[-(i-1)]
				znew = pts[i+1][0]
				if znew.size == 1: znew = pts[-(i-1)]
				pairs.append( zip(xnew,ynew,znew) )
				plt.cla()
				for j in (range( len(pairs) )):
					a = 0.0
					if j > ( len(pairs) - 15 ): 
						a = scipy.absolute( 1 - ( len(pairs) - j ) / float(len(pairs)) )
						if j == 0: a = 1.0/(i*2+1)
						if j > 15: a = scipy.absolute( 1 - (len(pairs) - 15 - j) / 15.0 )
					ax.plot(*pairs[j], c='black', alpha=a )
			ax.scatter(2,1,4,color='red',marker='o',s=40)

		if fname == None:
			plt.show()
		else:
			fig.savefig(fname)
		plt.close()

if __name__ == '__main__':

	import matplotlib.pyplot as plt
	
	alpha = scipy.array([10,200])
	center = scipy.array([2,1])
	x = scipy.matrix([3,3])
	# x = scipy.random.randint(-3,3,(4,2))
	sfunc = step_size(0.9,1)

	para = parabola.ParabolaDir(alpha,center)

	sgd = SGD(afunc=para,x0=x,sfunc=sfunc)

	print sgd.getSoln()

	# sgd.reset()

	for i in range(200):
		sgd.nsteps(1)
		fname = 'vid5/sgd_q1_{0:03d}'.format(i)
		# ax = plt.gca()
		# sgd.plot(alphaMult=0.9)
