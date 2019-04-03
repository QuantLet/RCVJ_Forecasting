import pandas as pd
import os
import numpy as np
import datetime as dt


def parse_datetime(x):
    date_part = dt.datetime.strptime(x[0][0].split('_')[0], '%d%b%Y')
    time_part = list(map(lambda y: dt.datetime.strptime(y[0].split(' - ')[1], '%H:%M').replace(
        year=date_part.year,
        month=date_part.month,
        day=date_part.day), x[1:]))
    return time_part


root_dir = os.getcwd()
data_dir = root_dir + '/Data/'
outdata_dir = root_dir + '/Output/Data/'
outplot_dir = root_dir + '/Output/Plot/'

sp500_5 = pd.read_csv(data_dir + 'SP500_5min.csv')

# CLEAN the DATA
sp500_array = sp500_5.values
t = np.arange(0, 100, 5)
daily_length = 82
sp500_array_split = [sp500_array[i:i + daily_length] for i in np.arange(0, len(sp500_array), daily_length)]
date_list = list(map(lambda x: parse_datetime(x), sp500_array_split))

date_idx = np.reshape(date_list, (1, -1))[0]

sp500_5.dropna(axis=0, inplace=True)
sp500_5['Time'] = date_idx
sp500_5.set_index(keys='Time', inplace=True)
sp500_5.columns = ['close', 'change', 'open', 'high', 'low']

sp500_5.to_csv(data_dir + 'SP500_IDX_Clean.csv')
