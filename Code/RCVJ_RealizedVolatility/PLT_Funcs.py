import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate as uni_kde
from sklearn.model_selection import GridSearchCV

import statsmodels.tsa.api as smt
from Code.RCVJ_DataOperation.IO_Funcs import read_dyos_clean, lower_freq, read_clean_gemini
from Code.GlobalParams import outplot_dir

root_dir = os.getcwd()
data_dir = root_dir + '/Data/'
outdata_dir = root_dir + '/Output/Data/'


def plot_logreturn_hist(data_df, bins=None, hist_density=False, drop_zero=True, random_sample_size=None, replace=False,
                        **kwargs):
    if bins is None:
        bins = int(round(len(data_df) / 20, 0))
    if drop_zero:
        data_df = data_df[data_df.values != 0]
    if not random_sample_size is None:
        random_idx = np.random.choice(
            len(data_df), random_sample_size, replace=replace)
        data_df = data_df.iloc[random_idx]
    values = data_df.values
    fig = plt.figure(figsize=(15, 5))
    plt.hist(values, bins=bins, density=hist_density)

    if kwargs['xlabel']:
        plt.xlabel(kwargs['xlabel'])
    else:
        plt.xlabel('Log-return')

    plt.ylim([0, 10000])
    plt.xlim([-0.01, 0.01])
    plt.show()

    plt.tight_layout()
    return fig, values


def plot_RV_separation(return_df: pd.Series, rv: pd.Series, bpv: pd.Series, jump: pd.Series, **kwargs):
    # transform into square root realized volatility and bipower volatility
    # rv = np.sqrt(rv)
    # bpv = np.sqrt(bpv)
    # plot

    upper_bound = max(np.sqrt(rv).max(), np.sqrt(bpv).max(), np.sqrt(jump).max())
    left_bound = max(min(rv.index), min(bpv.index))
    right_bound = min(max(rv.index), max(bpv.index))

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(4, 1, 1)

    plt.plot(return_df, color='black', linestyle='-')
    # plt.plot(return_df.values, color='black', linestyle='-', markersize=3)
    # plt.xlim([dt.date(2015, 12, 31), dt.date(2018, 8, 22)])
    plt.xlim([left_bound, right_bound])
    if 'freq' in kwargs:
        plt.title(f'{kwargs["freq"]} Log-return')
    else:
        plt.title('Log-return')

    plt.subplot(4, 1, 2)
    # plt.plot(np.sqrt(rv), color='blue', linestyle='-', linesize=3)
    plt.bar(x=rv.index, height=np.sqrt(rv).values, color='blue', width=0.6)
    # plt.bar(x=rv.index, height=np.sqrt(rv['RV']).values, color='blue', width=0.6)
    plt.title(kwargs['title_rv'])
    # plt.title('$RV^{1/2}$')
    plt.ylim([0, upper_bound * 1.1])
    plt.xlim([left_bound, right_bound])

    # plt.hist(rv_crix, 100)
    plt.subplot(4, 1, 3)
    # plt.plot(np.sqrt(bpv), color='orange', linestyle='-', markersize=3)
    plt.bar(x=rv.index, height=np.sqrt(bpv).values, color='orange', width=0.6)
    plt.title(kwargs['title_bpv'])
    # plt.title('$BPV^{1/2}$')
    plt.ylim([0, upper_bound * 1.1])
    plt.xlim([left_bound, right_bound])

    plt.subplot(4, 1, 4)
    # plt.plot(np.sqrt(jump), color='red', linestyle='-', markersize=3)
    plt.bar(x=rv.index, height=np.sqrt(jump).values, color='red', width=0.6)
    plt.title(kwargs['title_jump'])
    # plt.title('$J^{1/2}$')
    plt.xlabel('Time', fontsize=16)
    plt.ylim([0, upper_bound * 1.1])
    plt.xlim([left_bound, right_bound])

    plt.tight_layout()
    plt.show()
    return fig


def plot_RV_separation_notimeframe(return_df, rv, bpv, jump, freq='5min'):
    fig = plt.figure(figsize=(15, 7))
    plt.subplot(4, 1, 1)
    plt.plot(return_df.values, color='black', linestyle='-')
    # plt.bar(x=range(len(return_df)),height=return_df.values, color='black')
    # plt.xlim([dt.date(2015, 12, 31), dt.date(2018, 7, 2)])
    plt.ylim([min(return_df.values) * 0.9, max(return_df.values) * 1.1])
    plt.title(f'{freq} Log-return')

    plt.subplot(4, 1, 2)
    plt.bar(x=range(len(rv)), height=np.sqrt(rv).values, color='blue', width=0.2)
    plt.title('$RV^{1/2}$')
    # plt.ylim([0, 0.05])

    # plt.hist(rv_crix, 100)
    plt.subplot(4, 1, 3)
    plt.bar(x=range(len(bpv)), height=np.sqrt(bpv).values, color='orange', width=0.2)
    plt.title('$BPV^{1/2}$')
    plt.ylim([0, 0.05])

    plt.subplot(4, 1, 4)
    plt.bar(x=range(len(jump)), height=np.sqrt(jump).values, color='red', width=0.2)
    plt.title('$J^{1/2}$')
    plt.xlabel('Time', fontsize=16)
    plt.ylim([0, 0.05])

    plt.tight_layout()
    plt.show()
    return fig


def plot_daily_vol(logreturn_daily):
    # ==================Plot daily volatility of bitcoin_hf
    logreturn_daily['month'] = list(map(lambda x: f'{x.year}-{x.month}', logreturn_daily.index))
    vol_daily = logreturn_daily.groupby(by=logreturn_daily['month'], axis=0).apply(lambda x: x.std())
    vol_daily.dropna(axis=0, inplace=True)
    plt.figure(figsize=(15, 5))
    plt.plot(vol_daily)
    plt.ylabel('Log-return Daily Volatility')
    plt.xlabel('Month')
    plt.xticks(fontsize=7)
    plt.show()
    plt.tight_layout()
    plt.savefig(outplot_dir + 'DailyVol.png', dpi=300)


def plot_5min_vol(logreturn_ts):
    # ==================Plot 5-min volatility of bitcoin_hf
    vol_5min = logreturn_ts.groupby(by=logreturn_ts.index.date, axis=0).apply(lambda x: x.std())

    plt.figure(figsize=(15, 5))
    plt.plot(vol_5min)
    plt.ylabel('Log-return 5-min Volatility')
    plt.xlabel('Time')
    # plt.xticks(fontsize=7)
    # plt.show()
    plt.tight_layout()
    plt.savefig(outplot_dir + '5-min_Vol.png', dpi=300)


class PlotKernelDensityEstimator:
    """
    A object for plotting a series of KDE with given bandwidths and kernel functions
    """

    def __init__(self, data_points):
        if isinstance(data_points, list):
            data_points = np.asarray(data_points)
        self.data_points = data_points
        # Default parameters
        self.kernel = 'epanechnikov'
        self.x_grid = np.linspace(min(data_points), max(data_points), int(len(data_points) / 10))
        # self.x_plot = np.linspace(0, 1, 1000)
        # self.file_name = file_name

    def bandwidth_search(self, method):
        # x_grid = np.linspace(min(self.data_points), max(self.data_points), int(len(self.data_points)/10))
        print('Searching Optimal Bandwidth...')
        if method == 'gridsearch':
            grid = GridSearchCV(KernelDensity(),
                                {
                                    'bandwidth': self.x_grid,
                                },
                                cv=5)
            grid.fit(self.data_points.reshape(-1, 1))
            self.band_width = grid.best_params_['bandwidth']
        elif method == 'silverman':
            std = self.data_points.std()
            n = len(self.data_points)
            self.band_width = 1.06 * std * np.power(n, -1/5)
        return self.band_width

    def pdf_calcualtion(self, **kwargs):
        if 'bandwidth' in kwargs:
            self.bandwidth = kwargs['bandwidth']
        else:
            self.bandwidth = self.bandwidth_search(kwargs['method'])
            print(f"Bandwidth search method: {kwargs['method']}")
        kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(X=self.data_points.reshape(-1, 1))
        self.pdf = np.exp(kde.score_samples(self.x_grid.reshape(-1, 1)))
        # self.log_densities[f'{bandwidth}'] = log_dens
        return self.pdf

    def plot_curve_hist_kde(self, bin_num=None, hist_density=True, bandwidth=None, method='silverman'):
        if bandwidth is None:
            self.pdf_calcualtion(method=method)
        else:
            self.pdf_calcualtion(bandwidth=bandwidth)
        fig = plt.figure(figsize=(15, 7))
        plt.hist(self.data_points, bins=bin_num, density=hist_density)
        plt.plot(self.x_grid, self.pdf, '-')
        plt.title(f'Kernel Estimation')
        return fig

    # def save_plot(self, dir, num_bin, hist_density=True, refresh=True):
    #     self.bandwidth_search()
    #     self.log_density_estimate()
    #     plot_exist = os.path.exists(dir + f'/{self.file_name}_{self.kernel}_{self.band_width}.png')
    #     if not plot_exist or refresh:
    #         self.plot_curve_hist_kde(log_dens=self., bin_num=num_bin,
    #                                  hist_density=hist_density)
    #         self.fig.savefig(dir + f'/{self.file_name}_{self.kernel}_{self.band_width}.png', dpi=300)
    #         print(f'Plot: {self.file_name}_{self.kernel}_{self.band_width}')
    #         plt.close()
    #     else:
    #         print(f'{self.file_name}_{self.kernel}_{self.band_width}.png: Existed!')


def plot_logreturn_hists(coin, freq, low_cut, high_cut, zero=True):
    coin_2018, coin_2017, coin_2016 = (
        pd.read_csv(data_dir + f'gemini_{coin}USD_{year}_1min.csv', index_col=1, parse_dates=True)
        for year in ('2018', '2017', '2016'))

    coin_full = pd.concat([coin_2016, coin_2017, coin_2018], axis=0)
    coin_full.sort_index(ascending=True, inplace=True)

    # plt.figure(figsize=(16, 5))
    # plt.plot(coin_full['Close'])

    logprice = np.log(coin_full['Close'])
    logreturn = logprice - logprice.shift(1)
    logreturn.dropna(axis=0, inplace=True)

    logreturn_hf = lower_freq(logreturn, freq=freq, coin=coin)

    if not zero:
        logreturn_hf = logreturn_hf[~(logreturn_hf[f'{coin}_{freq}'] == 0)]

    high_thrsh = np.percentile(logreturn_hf[f'{coin}_{freq}'], high_cut)
    low_thrsh = np.percentile(logreturn_hf[f'{coin}_{freq}'], low_cut)

    logreturn_hf = logreturn_hf[(logreturn_hf[f'{coin}_{freq}'] <= high_thrsh) &
                                (logreturn_hf[f'{coin}_{freq}'] >= low_thrsh)]

    values = logreturn_hf.values
    bins = int(round(len(values) / 20, 0))
    hist_density = False
    fig = plt.figure(figsize=(15, 5))
    hist_info = plt.hist(values, bins=bins, density=hist_density)
    plt.xlabel('Log-return')
    # plt.ylim([0, 6000])
    # plt.xlim([-0.00020, 0.00020])
    # plt.show()

    fig.savefig(outplot_dir + f'/LogreturnHisto/Gemini_{coin}_{freq}_[{low_cut}_{high_cut}]_{zero}.png', dpi=300)
    plt.close()
    return hist_info


def plot_price_return_volume(time_index, price:pd.Series, logreturn:pd.Series, volume, high_count_value):
    # fig = plt.subplots(2,1, figsize=(15,6))
    fig = plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(2, 1, 1)
    t = pd.to_datetime(time_index)
    # a, = ax1.plot(t, price.values, 'b-', linewidth=2)
    a, = ax1.plot(t, price.values, 'b-', linewidth=2)

    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(price.columns[0], fontsize=16)
    ax1.tick_params('y')

    ax2 = ax1.twinx()
    b, = ax2.plot(t, logreturn.values, 'r-', linewidth=2, label=logreturn.name)
    ax2.plot(t, [high_count_value] * len(t), 'k--', linewidth=2)
    ax2.set_ylabel(logreturn.name, fontsize=16)
    ax2.tick_params('y')

    # Add stripe
    high_timestamp = logreturn[logreturn == high_count_value].index
    for timestamp in high_timestamp:
        ax2.axvline(x=timestamp, linewidth=2, color='green')

    plt.title(f'Intraday Close Price, Log-Return, {t.date[0]}, {high_count_value}')

    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(t, volume.values)
    ax3.set_ylabel('Volume', fontsize=16)
    ax3.set_xlabel('Date', fontsize=16)

    for timestamp in high_timestamp:
        ax3.axvline(x=timestamp, linewidth=2, color='green')

    """
    if data1.max() > data2.max():
        high_mark = data1.max()
    else:
        high_mark = data2.max()
    if data1.min() < data2.min():
        low_mark = data1.min()
    else:
        low_mark = data2.min()
    ylim(low_mark * 1.1, high_mark * 1.1)
    """
    plt.tight_layout()

    # p = [a,b]
    # ax1.legend(p, [p_.get_label() for p_ in p],
    #            loc='upper right', fontsize=13)
    # plt.show()
    return fig


def plot_acf(y, lags=None, title=None):
    fig = plt.figure(figsize=(20, 5))
    acf_all = plt.subplot(2, 1, 1)
    smt.graphics.plot_acf(y, lags=None, ax=acf_all, unbiased=True)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title(f'ACF plot of: {title}')
    acf_lags = plt.subplot(2, 1, 2)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_lags, unbiased=True)
    plt.xlabel('Time Lag', fontsize=18)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title('')
    plt.tight_layout()
    # plt.savefig(name + '.pdf', dpi=300)
    return fig


def plot_highest_return_day(coin='BTC', freq='5min'):
    """
    Plot the price movement of a given coin
    :param coin: coin's name, only BTC and ETH for now
    :param freq:
    :return:
    """

    # logreturn from original file
    # =============================

    coin_G = f'gemini_{coin}'
    coin_D = coin

    origin_data_G = read_clean_gemini(coin_G, freq, False, 0)
    origin_daily_D, _default = read_dyos_clean(coin_D, 0)
    origin_logreturn_G = origin_data_G['log_return']
    origin_daily_logreturn_G = origin_logreturn_G.groupby(by=origin_logreturn_G.index.date, axis=0, sort=True).apply(
        lambda x: x.sum())
    max_date = origin_daily_logreturn_G[origin_daily_logreturn_G == max(origin_daily_logreturn_G)]

    price_G = origin_data_G['Close']
    price_D = _default['close']

    max_price_btcG = price_G[price_G.index.date == max_date.index[0]]
    max_price_btcD = price_D[price_D.index.date == max_date.index[0]]

    plt.figure(figsize=(15, 7))
    plt.plot(max_price_btcG.index, max_price_btcG.values, color='b', linestyle='-', label='BTC-G')
    plt.plot(max_price_btcD.index, max_price_btcD.values, color='red', linestyle=(0, (1, 1)), label='BTC-D')
    plt.xlabel(f'Time ({max_date.index[0]})', fontsize=17)
    plt.xticks(fontsize=14)
    plt.ylabel('Price (U.S Dollar)', fontsize=17)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outplot_dir + f'{coin}_highest_return_day.png', dpi=300)
