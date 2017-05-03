import pandas
import scipy
import csv

csvfile = 'region-{0}.csv'

regions = [chr(i) for i in range(ord('a'),ord('p')+1)]

regions.remove('e')

first_year = 1933

all_years = scipy.arange(first_year,2018)
year_list = [str(e) for e in all_years]
year_list.insert(0,'Region')

with open('reservoir_annual_pfull_summary.csv','wb') as f:
	w = csv.writer(f)
	w.writerow(year_list)

	for l in regions:
		df = pandas.read_csv(csvfile.format(l),skiprows=29,usecols=['date','percent_full'])

		df['year'] = pandas.to_datetime(df['date']).dt.year
		years = list(df.year)
		year_dif = years[0] - first_year
		blanks = ['' for x in range(year_dif)]
		avg_pfull = list(df.groupby('year')['percent_full'].mean())
		merge = blanks + list(avg_pfull)
		merge.insert(0,l.capitalize())
		w.writerow(merge)