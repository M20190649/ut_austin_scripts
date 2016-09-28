import scipy
import matplotlib
import pylab
import matplotlib.patches

pylab.clf()
ax = pylab.gca()
for x in scipy.arange(10):
    for y in scipy.arange(10):
        anarrow = matplotlib.patches.Arrow(x,y,scipy.sin(scipy.pi*(x-5)/10.0),scipy.cos(scipy.pi*(x-5)/10.0),0.5,lw=2,fc='green',ec='red')
        ax.add_patch(anarrow)
pylab.xlim(-1,10)
pylab.ylim(-1,10)
pylab.show()