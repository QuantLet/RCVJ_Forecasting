import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime as dt


def round_time_values(row_values):
    if row_values['floored_time_diff'] <= row_values['ceiled_time_diff']:
        return row_values['floored_time']
    elif row_values['floored_time_diff'] > row_values['ceiled_time_diff']:
        return row_values['ceiled_time']

def clean_crix():
    root_path = os.getcwd()
    data_path = root_path + '/Data/'

    hf_ak = pd.read_csv(data_path + 'HF.csv', delimiter=';', index_col=[0])
    # Encode timestamp into New York City time, UTC-5
    nyc_tz = dt.timezone(dt.timedelta(hours=-5))
    date_index = [dt.datetime.fromtimestamp(date, tz=nyc_tz) for date in hf_ak.index]
    date_index_round = [dt.datetime(date.year, date.month, date.day, date.hour, date.minute, 0) for date in date_index]

    hf_ak['date'] = date_index_round
    hf_ak.set_index('date', inplace=True, drop=True)
    hf_ak = hf_ak[~ (hf_ak.values == hf_ak.max().values)]
    hf_ak = hf_ak[~ (hf_ak.values == hf_ak.max().values)]

    hf_ak.describe()

    plt.figure(figsize=(15, 6))
    plt.plot(hf_ak)
    # plt.savefig('dropped_plot.jpg', dpi=300)

    hf_ak.sort_index(axis=0, ascending=True, inplace=True)

    # Round index to 5 min to closest 5-min
    hf_ak['floored_time'] = [time.floor('5min') for time in hf_ak.index]
    hf_ak['ceiled_time'] = [time.ceil('5min') for time in hf_ak.index]
    hf_ak['floored_time_diff'] = hf_ak.index - hf_ak['floored_time']
    hf_ak['ceiled_time_diff'] = hf_ak['ceiled_time'] - hf_ak.index

    hf_ak['rounded_time'] = [round_time_values(row) for id, row in hf_ak.iterrows()]

    test_tdf = hf_ak[hf_ak['floored_time_diff'] != dt.timedelta(minutes=0)]

    hf_ak.set_index(keys='rounded_time', inplace=True, drop=True)

    # Drop duplicated entries and take the first one
    hf_ak_dropped = hf_ak[~ hf_ak.index.duplicated(keep='first')]
    plt.figure(figsize=(15, 6))
    plt.plot(hf_ak_dropped['price'])
    plt.plot(hf_ak['price'])
    hf_ak_dropped.drop(labels=['floored_time', 'ceiled_time', 'floored_time_diff', 'ceiled_time_diff'], inplace=True,
                       axis=1)

    # Check daily periods larger or equal to 280
    hf_ak_dropped_grouped = hf_ak_dropped.groupby(by=hf_ak_dropped.index.date, axis=0).count()

    # hf_ak_dropped_grouped = hf_ak.groupby(by=hf_ak.index.date, axis=0).count()

    hf_ak_288periods = hf_ak_dropped_grouped[hf_ak_dropped_grouped.values == 288]
    plt.figure(figsize=(15, 6))
    plt.plot(hf_ak_288periods)

    # Take only the days with more than 280 periods data
    hf_ak_quality = pd.DataFrame()
    hf_ak_grouped = hf_ak_dropped.groupby(by=hf_ak_dropped.index.date, axis=0)
    for num, data in hf_ak_grouped:
        print(num, data)
        if len(data) == 288:
            hf_ak_quality = pd.concat([hf_ak_quality, data], axis=0)
        else:
            pass

    # Calculate time difference in each day, see if they are all 5-min frequency
    hf_ak_quality['time'] = hf_ak_quality.index
    hf_ak_quality_timediff_daily = hf_ak_quality.groupby(by=hf_ak_quality.index.date, axis=0).apply(
        lambda x: x['time'] - x.shift(1)['time'])
    hf_ak_quality_timediff_daily_bad_freq = hf_ak_quality_timediff_daily[
        hf_ak_quality_timediff_daily != dt.timedelta(minutes=5)]
    hf_ak_quality_timediff_daily_bad_freq = hf_ak_quality_timediff_daily_bad_freq[
        ~ hf_ak_quality_timediff_daily_bad_freq.isnull()]
    # all 5-min frequency

    hf_ak_quality.drop(labels=['time'], axis=1, inplace=True)
    plt.figure(figsize=(15, 6))
    plt.plot(hf_ak_quality)

    return hf_ak_quality

