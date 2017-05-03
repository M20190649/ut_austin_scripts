import os
from math import atan2
import sys

parentfolder = "/home/paul/mystuff/school/spring2016/wrpm"

lakefolder = "/lakeslopes"

directory = parentfolder + lakefolder

output = "lakeslope_analysis"

extension = ".csv"

maxslopes = []

avgslopes = []

filenames = os.listdir(directory) # Names of all files in directory

parameters = []

outputfiles = []

# Collect parameter names
for file in filenames: 
	file = file[-(2+len(extension)):-len(extension)]
	if file in parameters: 
		continue
	parameters.append(file)

# Create output filenames: 
for file in filenames: 
	for p in range(len(parameters)):
		if (output + "_" + parameters[p] + extension) in outputfiles: 
			continue
		outputfiles.append(output + "_" + parameters[p] + extension)

# Delete file if it already exists
for file in outputfiles: 
	for p in range(len(parameters)): 
		if (output + "_" + parameters[p] + extension) in filenames:
			os.remove(directory + "/" + output + "_" + parameters[p] + extension)
			filenames = os.listdir(directory)
			print "Existing file %s deleted" % (output + parameters[p] + extension)

# Average function
def average(numbers): 
	total = 0.0
	for n in range(len(numbers)):
		floatnum = float(numbers[n])
		total += floatnum
	average = total / len(numbers)
	return average

def lakenumber(filename):
	dotspot = 0 
	lakenumber = ""
	for x in range(len(filename)):
		if filename[x] == "_": 
			dashspot = x
			lakenumber = filename[4:dashspot]
	return lakenumber

# Open all files in parent folder
for file in filenames: 
	x_values = []
	y_values = []
	slopes = []
	# Read lines in file
	with open(directory + "/" + file, "r") as current: 
		lines = current.readlines()
		# Save x and y variables to x_values and y_values lists
		for i in (range(1,len(lines)-1)): 
			xvar = lines[i].split()[0]
			x_values.append(xvar)
			yvar = lines[i].split()[1]
			y_values.append(yvar)
	# Calculate and append all slopes (in degrees) to slopes list
	for j in range(len(x_values)):
		if j != 0:
			rise = abs(float(y_values[j]) - float(y_values[j-1]))
			run = abs(float(x_values[j]) - float(x_values[j-1]))
			slopedeg = atan2(rise, run) * 100
			if slopedeg > 90:
				slopedeg -= 90
			slopes.append(slopedeg)
	# Calculate max slope and append to maxslopes list
	maxslope = float(max(slopes))
	maxslopes.append(maxslope)

	# Calculate average slope and append to avgslopes list
	avgslope = float(average(slopes))
	avgslopes.append(avgslope)

# Write all maxslope and avgslope values to <output>.csv
for p in range(len(parameters)): 
	with open(parentfolder + "/" + outputfiles[p], "w+") as final: 
		print parentfolder + "/" + outputfiles[p]
		final.write("File Name, Parameter, Maximum Slope (deg), Average Slope (deg)\n")
		for z in range(len(filenames)):
			filename = filenames[z]
			number = lakenumber(filename)
			maxslope = str("{0:.3f}".format(maxslopes[z]))
			avgslope = str("{0:.3f}".format(avgslopes[z]))
			if filename[-(2+len(extension)):-len(extension)] == parameters[p]:
				final.write(filename + ", " + number + ", " + maxslope + ", " + avgslope + "\n")
	print "New file successfully created at", (parentfolder + "\\" + outputfiles[p])