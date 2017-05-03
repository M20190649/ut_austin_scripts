import pandas
import matplotlib.pyplot as plt
import scipy

df = pandas.read_csv('frequency_summary.csv')

print df

fig = plt.figure()

width = 0.35
ind = scipy.arange(len(df['Classification'].values))
plt.bar(ind,df['Percent'].values)
plt.xticks(ind + width / 2, df['Classification'].values,rotation='60')
plt.tight_layout()
# plt.savefig('classification_frequency.png')
plt.show()