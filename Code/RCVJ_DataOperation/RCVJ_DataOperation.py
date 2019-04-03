"""
This file is not for execution! Please refer to other code files
"""

import datetime as dt
import os
import numpy as np
import pandas as pd
from Code.GlobalParams import data_dir


def read_coin_return(coin, freq, refresh, tz_lag):
    """
    Given coin name and frequency, read log-returns
    :param coin:
    :param freq:
    :param refresh:
    :param tz_lag:
    :return:
    """
    try:
        # if not backup_data:
        logreturn_hf, _default = read_dyos_clean(coin, tz_lag)
        if freq != '5min':
            logreturn_hf = lower_freq(ts_df=logreturn_hf, freq=freq)

    except FileNotFoundError:
        coin_data = read_clean_gemini(coin=coin, freq=freq, refresh=refresh, tz_lag=tz_lag)
        logreturn_hf = coin_data['log_return'].to_frame()
        # logreturn = logreturn_hf.copy(deep=True)
    return logreturn_hf


def read_dyos_clean(coin, tz_lag=0):
    root_dir = os.getcwd()
    data_dir = root_dir + '/Data/'

    # Calculate the high frequency return in each day since we don't have
    # Access the New HF data from Dynamica company
    try:
        close_price = pd.read_csv(data_dir + f'{coin}_USDT.csv', index_col=0, parse_dates=True)
    except:
        close_price = pd.read_csv(data_dir + f'/{coin}.csv', index_col=0, parse_dates=True)
    timezone = dt.timezone(dt.timedelta(hours=tz_lag))
    # close_price['Time'] = [dt.datetime.fromtimestamp(float(timestamp), tz=timezone) for timestamp in close_price['timestamp']]
    # close_price.set_index(keys='Time', drop=True, inplace=True)
    close_price = close_price.tz_convert(tz=timezone)
    close_price = close_price.tz_convert(tz=None)

    logprice_ts = np.log(close_price['close']).to_frame()

    logreturn = logprice_ts - logprice_ts.shift(1)

    logreturn.dropna(axis=0, inplace=True)
    logreturn.columns = [f'log_return']

    return logreturn, close_price


def lower_freq(ts_df, freq='10min'):
    if isinstance(ts_df, pd.Series):
        ts_df = ts_df.to_frame()
    lower_freq = ts_df.copy(deep=True)
    lower_freq['Time'] = lower_freq.index.ceil(freq)
    lower_freq.set_index(keys='Time', drop=True, inplace=True)
    lower_freq_agg = lower_freq.groupby(by=lower_freq.index, axis=0).apply(lambda x: sum(x[ts_df.columns[0]]))
    lower_freq_agg = lower_freq_agg.to_frame()
    lower_freq_agg.columns = [f'log_return']
    return lower_freq_agg


def read_clean_gemini(coin='gemini_BTC', freq='5min', refresh=False, tz_lag=0):
    """
    Depreciate choosing different timezone for now
    Set GMT +0 as default timezone
    :param coin:
    :param freq:
    :param refresh:
    :param tz_lag:
    :return:
    """
    timezone = dt.timezone(dt.timedelta(hours=tz_lag))

    if os.path.exists(data_dir + f'{coin}_{freq}.csv') and not refresh:
        coin_full = pd.read_csv(data_dir + f'{coin}_{freq}.csv', index_col=0, parse_dates=True)
        # coin_full = coin_full.tz_localize('UTC').tz_convert(timezone)
        # coin_full = coin_full.tz_convert(timezone)


    else:
        coin_2019, coin_2018, coin_2017, coin_2016 = (
            pd.read_csv(data_dir + f'{coin}USD_{year}_1min.csv', index_col=1, parse_dates=True)
            for year in ('2019', '2018', '2017', '2016'))

        coin_full = pd.concat([coin_2016, coin_2017, coin_2018, coin_2019], axis=0)
        coin_full.sort_index(ascending=True, inplace=True)

        coin_full_daily_sample_num = coin_full.groupby(coin_full.index.date, axis=0).apply(lambda  x: len(x))
        bad_sample_date = coin_full_daily_sample_num[coin_full_daily_sample_num <=1400].index

        for date in bad_sample_date:
            coin_full = coin_full[~(coin_full.index.date == date)]  # Weird data on 2018.8.23, drop them

        # coin_2018['Unix Timestamp'] = [stamp if len(str(stamp)) ==10 else int(stamp/1000) for stamp in coin_2018['Unix Timestamp']]

        # coin_full['timestamp'] = [float(dt.datetime.timestamp(datetime)) for datetime in coin_full.index]

        # coin_full['Time'] = [dt.datetime.fromtimestamp(timestamp, tz=timezone) for timestamp in
        #                      coin_full['Unix Timestamp']]
        # coin_full.set_index('Time', inplace=True)

        # plt.figure(figsize=(16, 5))
        # plt.plot(coin_full['Volume'])

        # accumulate trading volume
        volume = coin_full.Volume.to_frame()
        volume['Time'] = volume.index.ceil(freq)
        volume.set_index('Time', inplace=True)
        volume = volume.groupby(by=volume.index, axis=0).apply(lambda x: x.sum(axis=0))
        # volume = volume[volume.index.date >=dt.date(2015,12,31)]

        # select 5-min samples of close, open, high, low
        prices = coin_full[['Open', 'High', 'Low', 'Close']]
        prices = prices[np.mod(prices.index.minute, int(freq.split('min')[0])) == 0]

        # Concat
        coin_full = pd.concat([prices, volume], axis=1)
        coin_full.dropna(inplace=True)

        # Log-return
        logprice = np.log(coin_full['Close'])
        coin_full['log_return'] = logprice - logprice.shift(1)
        coin_full.dropna(axis=0, inplace=True)

        coin_full.to_csv(data_dir + f'{coin}_{freq}.csv')
    return coin_full


def panel_to_multits(panel_df, ind_col_name, variable_name):
    """
    Give a panel matrix, individual column name, variable name.
    Return multiple time series for one variable
    :param panel_df: panel matrix
    :param ind_col_name: Individual column name
    :param variable_name: a random variable name
    :return:
    """
    panel_tf = panel_df.loc[:, [ind_col_name, variable_name]]
    individual = panel_df[ind_col_name].unique()
    ts_df = pd.DataFrame(columns=individual)
    for ind in individual:
        ts_ind = panel_tf[panel_tf[ind_col_name] == ind][variable_name]
        ts_ind.name = ind
        ts_df[ind] = ts_ind
    return ts_df


def read_indices_full_RV():
    ox_ind_rv = pd.read_csv(data_dir + 'ox_index_rv.csv', index_col=0, parse_dates=True)

    # Transform panel data to RV ts, open price ts and close price ts
    rv_5 = panel_to_multits(ox_ind_rv, ind_col_name='Symbol', variable_name='rv5')
    open_price = panel_to_multits(ox_ind_rv, ind_col_name='Symbol', variable_name='open_price')
    close_price = panel_to_multits(ox_ind_rv, ind_col_name='Symbol', variable_name='close_price')

    """
    Correct overnight and holiday non-trading bias by add overnight rv to the next trading day 
    """
    overnight_rv = np.square(np.log(close_price) - np.log(open_price.shift(1)))
    overnight_rv.fillna(0, inplace=True)

    rv_overnight_5 = rv_5 + overnight_rv
    rv_overnight_5_annualized = rv_overnight_5 * 365
    return rv_overnight_5_annualized


def read_indices_trading_rv_bpv():
    ox_ind_data = pd.read_csv(data_dir + 'ox_index_rv.csv', index_col=0, parse_dates=True)
    rv_5 = panel_to_multits(ox_ind_data, ind_col_name='Symbol', variable_name='rv5')
    bpv_5 = panel_to_multits(ox_ind_data, ind_col_name='Symbol', variable_name='bv')
    jump = rv_5 - bpv_5
    return rv_5, bpv_5, jump
