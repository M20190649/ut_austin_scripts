import pandas
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import Counter

csvtemp = 'region_{0}.csv'

letters = [chr(i) for i in range(ord('a'),ord('p')+1)]

fig,ax = plt.subplots()

matplotlib.rcParams.update({'font.size': 20}) # Change font size everywhere, mainly for point labels

producing_water = []
not_implemented = []

for letter in letters:
	csvfile = csvtemp.format(letter)
	if os.path.isfile(csvfile) and letter != 'p':
		print csvfile
		df = pandas.read_csv(csvfile,usecols=['Infrastructure_Type','Summary','Implementation_Status'])
		df = df.dropna()
		providing = df[df['Implementation_Status'] == 'Providing Water']
		failing = df[df['Implementation_Status'] == 'Not Functional']
		
		producing_water = producing_water + list(providing.Infrastructure_Type.values)
		not_implemented = not_implemented + list(failing.Infrastructure_Type.values)
		print Counter(providing.Infrastructure_Type.values)
		print Counter(failing.Infrastructure_Type.values)