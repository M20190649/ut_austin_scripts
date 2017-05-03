import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x = scipy.array([1,2,3,4,5,6,7,8,9,10,11])
y = scipy.array([21,16.5,13,11,9.5,8.5,7.5,7,6.5,6.3,6.2])

def func(x, a, b, c): 
	return a * scipy.exp(-b * x) + c

popt, pcov = curve_fit(func, x, y)

plt.scatter(x,y,c='b')
plt.plot(x, func(x, *popt), c='r')

plt.show()