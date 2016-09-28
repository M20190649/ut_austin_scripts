
import random

point_count = [45,21,6,6,8,5,4,10,5,8,4,5,21,2,3,9,7,7,6,7,4,15,6,9,6,4,31,30,3,9,45,30,4,3,11,4,20,3,9,4,6,5,8,13,52,12,6,4,5,6,15,13,9,4,4,7,10,14,22,5,13,7,47,8,11,24,9,5,13,8,7,14,17,6,5,5,10,6,9,8,4,22,12,15,6,47,9,34,102,6,9,19,7,1,4,10,20,24,8,6,4,10,22,11,18,5,8,25,20,10,5,5,9,14,4,10,4,9,6,5,6,13,4,11,38,6,11,7,16,20,8,10,9,4,4,12,13,13,11,13,11,14,19,16,18,12,9,7,4,6,12,13,7,6,6,5,4.5,4,4,4,8,9,13,11,16,12,19,36,10,22,19,19,5,3,4,4,6,3,8,4,4,3,5,3,4]
numvars = len(point_count)
numruns = 100
samplesizes = []
finalerrors = []

def average(initlist): 
	sum = 0
	count = len(initlist)
	for i in initlist: 
		sum += i
	average = sum / count
	return average

def median(initlist):
	l = sorted(initlist)
	length = len(l)
	half_length = length/2
	median = 0
	if length % 2 != 0: 
		median += float(l[half_length])
	else: 
		median += float(l[half_length] + l[half_length-1])/2
	return median

def medianlist(initlist, numvars, numruns): 
	medlist = []
	current = []
	for _ in range(numruns): 
		nowlist = initlist[:]
		for i in range(numvars):
			rand = random.choice(nowlist)
			current.append(rand)
			nowlist.remove(rand)
		medlist.append(median(current))
	return medlist

def error(median, medianlist):
	error = abs((median-average(medianlist))/median)*100
	return error

for i in range(1,numvars): 
	med = median(point_count)
	medlist = medianlist(point_count, i, numruns)
	samplesizes.append(str(i))
	finalerrors.append(float(error(med,medlist)))
	print "No. Samples %s done." % i

with open("C:\d50_10000.csv",  "w") as fp:
        fp.write("No_Samples, Percent_Error\n")
        for i in range(0,numvars-1):
                fp.write("%s, %s\n" % (samplesizes[i],finalerrors[i]))

print "Done"
