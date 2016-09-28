# D50 = median of samples
# Determine how many samples are needed to determine D50 +/- 50%

import random

point_count = [45,21,6,6,8,5,4,10,5,8,4,5,21,2,3,9,7,7,6,7,4,15,6,9,6,4,31,30,3,9,45,30,4,3,11,4,20,3,9,4,6,5,8,13,52,12,6,4,5,6,15,13,9,4,4,7,10,14,22,5,13,7,47,8,11,24,9,5,13,8,7,14,17,6,5,5,10,6,9,8,4,22,12,15,6,47,9,34,102,6,9,19,7,1,4,10,20,24,8,6,4,10,22,11,18,5,8,25,20,10,5,5,9,14,4,10,4,9,6,5,6,13,4,11,38,6,11,7,16,20,8,10,9,4,4,12,13,13,11,13,11,14,19,16,18,12,9,7,4,6,12,13,7,6,6,5,4.5,4,4,4,8,9,13,11,16,12,19,36,10,22,19,19,5,3,4,4,6,3,8,4,4,3,5,3,4]

numvars = 20

runsum = 0
runavg = 0
numruns = range(1)

def median(initlist):
	l = sorted(initlist)
	length = len(l)
	half_length = length/2
	median = 0
	if length % 2 != 0: 
		# print "if statement"
		median += float(l[half_length])
	else: 
		# print "else"
		median += float(l[half_length] + l[half_length-1])/2
	return median

def d50_okay(initlist, n): 
	nowlist = initlist[:]
	current = []
	perc_dif = []
	sum_perc = 0
	avg_perc = 0
	for i in range(n): 
		rand = random.choice(nowlist) 
		current.append(rand) 
		nowlist.remove(rand)
	med = median(current)
	for x in range(len(current)): 
		abs_cur = abs(med - current[x]) 
		perc_dif.append(abs_cur/med) 
	for y in range(len(perc_dif)):
		sum_perc += perc_dif[y]
		print "Value %.2f = %.2f" % (current[y], perc_dif[y])
	avg_perc = (sum_perc / len(perc_dif))*100

	return avg_perc

for _ in numruns: 
	thisrun = d50_okay(point_count, numvars)
	runsum += thisrun

runavg = runsum/len(numruns)

print "Overall average percent difference: %.2f" % runavg