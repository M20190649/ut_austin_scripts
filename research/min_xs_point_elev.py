import pandas
import scipy

file = '/media/paul/pman/pjruess/research/onionck/cross_sections_onionck/'\
	'hecras_xspoint.csv'

df = pandas.read_csv(file,usecols=['RIVERSTATION','ELEVATION'],dtype=str)

min_elevs = {}

stations = df['RIVERSTATION'].unique()

for rs in stations:
	elev_vals = df[df.RIVERSTATION==rs].ELEVATION.values
	min_elev = scipy.amin(elev_vals)
	min_elevs[rs] = [min_elev]

elev_df = pandas.DataFrame.from_dict(min_elevs,orient='index')

elev_df.reset_index(inplace=True)

elev_df.columns = ['RiverStation','MinimumElevation']

dest = '/media/paul/pman/pjruess/research/onionck/cross_sections_onionck/rs_min_elev.csv'

elev_df.to_csv(dest)