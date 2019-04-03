import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt

root_path = os.getcwd()
data_path = root_path + '/Data/'

hf_ak = pd.read_csv(data_path + 'HF.csv', delimiter=';', index_col=[0])
date_index = [dt.datetime.fromtimestamp(date) for date in hf_ak.index]
hf_ak['date'] = date_index
hf_ak.set_index('date', inplace=True, drop=True)
hf_ak = hf_ak[~ (hf_ak.values == hf_ak.max().values)]
hf_ak = hf_ak[~ (hf_ak.values == hf_ak.max().values)]

hf_ak.describe()
plt.figure(figsize=(15,6))
plt.plot(hf_ak)
plt.savefig('dropped_plot.jpg', dpi=300)

hf_ak.sort_index(axis=0, ascending=True,inplace=True)


print(hf_ak)

days = set(hf_ak.index.date)
all_days = pd.date_range(start=min(days),end=max(days),freq='D')
all_days = set(date.to_datetime().date() for date in all_days)
days_miss = all_days.difference(days)

print(len(days_miss))

plt.figure(figsize=(15,6))
plt.hist(list(days_miss))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('miss_day_distribution.jpg', dpi=300)


hf_ak_grouped = hf_ak.groupby(by=hf_ak.index.date, axis=0).count()


daily_miss = hf_ak_grouped[hf_ak_grouped['price'] < 288]

dirty_days = hf_ak_grouped[hf_ak_grouped['price'] >288]

daily_miss_valuecount = daily_miss['price'].value_counts()

plt.hist(daily_miss)

print(len(daily_miss))

example = hf_ak[hf_ak.index.date == dt.date(2018,2,8)]









#======================================================================================
"""
hf_cj_1 = pd.read_csv(data_path + 'HFcrix.csv', delimiter=';', index_col=[0])
date_index = [dt.datetime.fromtimestamp(date) for date in hf_cj_1.index]
hf_cj_1['date'] = date_index
hf_cj_1.set_index('date', inplace=True, drop=True)
hf_cj_1 = hf_cj_1[~ (hf_cj_1.values == hf_cj_1.max().values)]

hf_cj_1.describe()

plt.plot(hf_cj_1)





hf_cj_2 = pd.read_csv(data_path + 'HFcrix2.csv')

hf_cj_2.dropna(axis=0, inplace=True)
hf_cj_2.set_index('time', inplace=True, drop=True)
hf_cj_2.drop(labels='date', inplace=True, axis=1)
hf_cj_2.sort_index(axis=0, ascending=True, inplace=True)
hf_cj_2 = hf_cj_2[~ (hf_cj_2.values == hf_cj_2.max().values)]
hf_cj_2.describe()

hf_cj_2.to_csv(data_path + 'jh_test.csv')

cj2_null = hf_cj_2[hf_cj_2['price'].isnull()]

plt.plot(hf_cj_2)



btceUSD = pd.read_csv(data_path + 'btceUSD.csv', header=None, index_col=0)
date_col = [dt.datetime.fromtimestamp(date) for date in btceUSD.index]
btceUSD['date'] = date_col
btceUSD.set_index('date', inplace=True, drop=True)
btceUSD.columns = ['bitcoin', 'USD']
btceUSD['bitcoin'].describe()
btceUSD['bitcoin'].plot()

"""

