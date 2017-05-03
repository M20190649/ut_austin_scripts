import pandas

csvfile = 'us_eia_energy_alldata.csv'

df = pandas.read_csv(csvfile,usecols=['MSN','StateCode','Year','Data'])

df = df[df.StateCode != 'AK'] # Remove Alaska
df = df[df.StateCode != 'DC'] # Remove District of Columbia
df = df[df.StateCode != 'HI'] # Remove Hawaii
df = df[df.StateCode != 'US'] # Remove United States Total

# msn = 'TETCB' # 'TETPB'
# title='Total_Energy_Consumed_Billion_BTU' # 'Total_Energy_Consumption_Per_Capita_Million_BTU'

# msn = 'TECPB'
# title = 'IEA_Commercial_Energy_Consumption_per_Capita_Million_BTU'

# msn = 'TECCB'
# title = 'IEA_Commercial_Energy_Consumption_Billion_BTU'

msn = 'TEICB'
title = 'IEA_Industrial_Energy_Consumption_Billion_BTU'

# msn = 'TEIPB'
# title = 'IEA_Industrial_Energy_Consumption_Per_Capita_Million_BTU'

# msn = 'TERCB'
# title = 'IEA_Residential_Energy_Consumption_Billion_BTU'

# msn = 'TERPB'
# title = 'IEA_Residential_Energy_Consumption_Per_Capita_Million_BTU'

year = 2010

df_msn = df[df['MSN'] == msn]

df_msn_year = df_msn[df_msn['Year'] == 2014]

# years = [year for year in df_msn.Year.unique()]

df_msn_year = df_msn_year.drop('Year', 1) # Remove Year column

df_msn_year = df_msn_year.drop('MSN', 1) # Remove MSN column

df_msn_year = df_msn_year.rename(columns={'Data':title}) # Improve header

# print df_msn_year

df_msn_year.to_csv('us_eia_{0}_{1}.csv'.format(msn.lower(),year), index=False)