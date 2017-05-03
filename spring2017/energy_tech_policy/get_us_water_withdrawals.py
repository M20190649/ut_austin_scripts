import pandas

csvfile = 'usgs_water_use_data_2010.csv'

# query = 'PS-Wtotl' # 'PS-WFrTo' # 'TO-WFrTo' # 'TO-WTotl'

query = 'DO-WFrTo' # Domestic, total self-supplied MGD (not including deliveries from Public Supply)

# query = 'IN-Wtotl' # Industrial total self-supplied MGD

# query = 'PC-WTotl' # Thermoelectric recirculation, total withdrawals MGD

# query = 'PO-WTotl' # Thermoelectric once-through, total withdrawals MGD

df = pandas.read_csv(csvfile,usecols=['STATE',query])

df = df[df.STATE != 'AK'] # Remove Alaska
df = df[df.STATE != 'DC'] # Remove District of Columbia
df = df[df.STATE != 'HI'] # Remove Hawaii
df = df[df.STATE != 'PR'] # Remove Puerto Rico
df = df[df.STATE != 'VI'] # Remove Virgin Islands

gb = df.groupby('STATE')[query].sum()

df_groupby = pandas.DataFrame(gb.reset_index())

# print df_groupby

df_groupby.to_csv('us_usgs_domestic_nodeliveries_total_withdrawals_mgpd.csv', index=False)