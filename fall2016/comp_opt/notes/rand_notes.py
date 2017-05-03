import pylab
import matplotlib.pyplot as plt
import scipy

"""Uniformly random directions"""

# 1000 random coordinates
ans = scipy.rand(1000,2)

# Normal dist
ans = scipy.randn(1000,2)


norms = scipy.sqrt((ans*ans).sum(axis=1))

ans = ans / norms.reshape(1000,1)

pylab.scatter(ans[:,0],ans[:,1])
plt.show()

# Chi-square: small area compared to number of points in area

"""Gradient Descent"""

# Gradient of point points towards direction that will occur the most --> approaches smaller and smaller values
# Stochastic GD has unknown gradient direction (rather than supplied). Random variable with expectation gradient. 