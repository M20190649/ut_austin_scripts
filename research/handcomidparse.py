from netCDF4 import Dataset
import pandas

idlookup = pandas.read_csv('streamgages.csv',usecols=['SOURCE_FEA','FLComID'])
idlookup['SOURCE_FEA'] = idlookup['SOURCE_FEA'].apply(lambda x: '{0:0>8}'.format(x))

handnc = Dataset('onionck/OnionCreek.nc', 'r')

def get_comid_indices(handnc,comids): 
	handc = handnc.variables['COMID']
	res = {}
	for i in range(len(handc)):
		if str(handc[i]) in comids and len(comids) > 0:
			print 'just got {0}'.format(handc[i])
			res[i]=handc[i]
			comids.remove(str(handc[i]))
		if len(comids) == 0:
			break
	return res

def write_to_file(res,filename):
	with open(filename,'w') as f:
		f.write('index,comid\n')
		for i in res:
			f.write('{0},{1}\n'.format(i,res[i]))


comids = []
for i in range(len(idlookup)):
	comids.append(str(idlookup.loc[i,'FLComID']))

print 'comids',comids

res = get_comid_indices(handnc,comids)

print res
# for i in res:
# 	print i
# 	print res[i]

write_to_file(res,'handnc_idx_test.csv')