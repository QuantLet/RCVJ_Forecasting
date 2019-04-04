[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **RCVJ_RealizedVolatility** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: RCVJ_RealizedVolatility

Published in: Realized Cryptocurrencies Volatility Forecasting with Jumps

Description: This quantlet includes two main code files. The first one named CAL_RV_BPV.py is the estimation of realized volatility, bipower volatility, threshold bipower volatility and separated jumps, threshold jumps. The second one named PLT_RV_BPV.py generates all related figures in the paper. Also some other auxiliary files that generate different statistics results.

Keywords: Realized volatility, Bipower volatility, Threshold Jump, Local variation estimation, Cryptocurrencies, Global Market Indices, ACF

Author: Junjie Hu

Submitted: 03.04.2019

```

### PYTHON Code
```python

"""
This file is not for execution! Please refer to other code files
"""

"""
CAL_RV_BPV.py
"""


import matplotlib as mpl

mpl.use('TKAgg')
from Code.DataOperation.IO_Funcs import read_coin_return
import pandas as pd
import datetime as dt
import itertools
from scipy.stats import norm
import scipy.special as ss
from Code.GlobalParams import *


def jump_test_statistics(rv, bpv_ct, tpv_ct):
    # print(rv)
    # print(ctbpv)
    # print(cttpv)
    if bpv_ct != 0:
        max_func = max(1, tpv_ct / (bpv_ct ** 2))
    else:
        max_func = 1
    return (rv - bpv_ct) / (rv * np.sqrt(((np.pi ** 2) / 4 + np.pi - 5) * max_func))


def upper_incomplete_gamma_func(gamma, cv):
    alpha = (gamma + 1) / 2
    x = (cv ** 2) / 2
    return ss.gamma(alpha) * ss.gammaincc(alpha, x)


def high_vol_smoother(theta, cv, gamma):
    # cv = 3
    # Gamma(1, x) = exp(-x)
    """
    When the square return r^2 is larger than the threshold theta, the expected value of r^gamma
    conditioning on r^2 > theta will be:
    (2*theta/cv^2)^(gamma/2) * upper_incomplete_Gamma((gamma+1)/2, cv^2/2) / 2N(-cv)sqrt(pi)
    ref: Corsi et al (2010)
    :param theta:
    :param cv:
    :param gamma:
    :return:
    """
    return ((((2 * theta) / (cv ** 2)) ** (gamma / 2)) * upper_incomplete_gamma_func(gamma, cv)) / (
            2 * norm.cdf(-cv) * np.sqrt(np.pi))


def mu_function(p):
    return 2 ** (p / 2) * ss.gamma((p + 1) / 2) / ss.gamma(1 / 2)


def st_norm_percentile(q):
    # q=0.9999
    return norm.ppf(q)


# def plot_significant_jump(z_test, sig_level, sig_ind):
#     test_z = z_test.copy(deep=True)
#     test_z = test_z.to_frame()
#     test_z['sig_level'] = [sig_level]*len(z_test)
#     test_z['sig_ind'] = sig_ind
#     plt.subplots(2,1,figsize=(15,5))
#     plt.subplot(2,1,1)
#     plt.plot(test_z['z'])
#     plt.plot(test_z['sig_level'], linewidth = 2, linestyle='--', color='red')
#     plt.title('Z-test')
#     plt.subplot(2,1,2)
#     # plt.plot(test_z['sig_ind'])
#     plt.plot(z_test['sig_jump'])
#     plt.title('Significant Jumps')
#     plt.xlabel('Date', fontsize=16)
#     plt.tight_layout()
#     plt.savefig(outplot_dir + 'rv_significant_jump.png', dpi=300)


def realized_volatility_separation(return_df, delta=1 / 288, alpha=0.9999, truncate_zero=False, annualized=True):
    if not isinstance(return_df, pd.DataFrame):
        return_df = return_df.to_frame()
    return_df.columns = ['log_return']

    # return_df = return_df[(return_df.index.date < dt.date(2018, 7, 1)) & (return_df.index.date > dt.date(2015, 12, 30))]

    # ========Calculate realized volatility
    rv = return_df.groupby(by=return_df.index.date, axis=0).apply(lambda x: sum(np.square(x['log_return']))).to_frame()
    rv.columns = ['RV']
    rv['date'] = rv.index
    rv.set_index(keys='date', drop=True, inplace=True)
    rv.dropna(axis=0, inplace=True)

    # ========Calculate Bipower variation
    # mu_1 = np.sqrt(2 / np.pi)
    abs_return = np.abs(return_df)
    mu_1 = mu_function(1)
    adj_square_return = abs_return.shift(1) * abs_return
    adj_square_return.dropna(axis=0, inplace=True)
    bpv = adj_square_return.groupby(by=adj_square_return.index.date, axis=0).apply(lambda x: x.sum(axis=0))
    # bpv = hf_quality_log_return_wthinday.groupby(by=hf_quality_log_return_wthinday.index.date, axis=0).apply(
    #     lambda x: np.nansum(x['return'].shift(1) * x['return'], axis=0)).to_frame()
    bpv = bpv * (mu_1 ** (-2))
    bpv.columns = ['BPV']
    bpv['date'] = bpv.index
    bpv.set_index(keys='date', drop=True, inplace=True)
    bpv.dropna(axis=0, inplace=True)

    # ========Calculate Tripower variation
    mu_43 = mu_function(4 / 3)
    adj_quart_return = (abs_return.shift(2) ** (4 / 3)) * (abs_return.shift(1) ** (4 / 3)) * (abs_return ** (4 / 3))
    adj_quart_return.dropna(axis=0, inplace=True)
    tpv = adj_quart_return.groupby(by=adj_quart_return.index.date, axis=0).apply(lambda x: x.sum(axis=0))
    tpv = tpv / ((delta) * (mu_43 ** 3))

    # concat rvs
    rvs = pd.concat([rv, bpv, tpv], axis=1)
    rvs.columns = ['RV', 'BPV', 'TPV']
    if annualized:
        rvs = rvs * 365

    # ========Calculate jump statistics, called z
    rvs['z'] = list(map(lambda rv, bpv, tpv: jump_test_statistics(rv, bpv, tpv), rvs['RV'], rvs['BPV'], rvs['TPV']))
    rvs['z'] = rvs['z'] / np.sqrt(delta)

    # ========Calculate jump component
    jump = rvs['RV'] - rvs['BPV']
    if truncate_zero:
        jump[jump < 0] = 0
    rvs['Jump_raw'] = jump.copy(deep=True)
    sig_level = norm.ppf(alpha)
    sig_ind = [1 if z_value > sig_level else 0 for z_value in rvs['z']]
    rvs[f'Jump_{alpha}'] = (jump) * sig_ind

    # ===Check with plot
    # test_z = rvs['z'].copy(deep=True)
    # test_z = test_z.to_frame()
    # test_z['sig_level'] = [sig_level]*len(rvs)
    # test_z['sig_ind'] = sig_ind
    # plt.subplots(2,1,figsize=(15,5))
    # plt.subplot(2,1,1)
    # plt.plot(test_z['z'])
    # plt.plot(test_z['sig_level'], linewidth = 2, linestyle='--', color='red')
    # plt.title('Z-test')
    # plt.subplot(2,1,2)
    # # plt.plot(test_z['sig_ind'])
    # plt.plot(rvs['sig_jump'])
    # plt.title('Significant Jumps')
    # plt.xlabel('Date', fontsize=16)
    # plt.tight_layout()
    # plt.savefig(outplot_dir + 'rv_significant_jump.png', dpi=300)
    # ===End Check
    return rvs


def local_variation_estimate(df, ii, cv):
    """
    Estimating local variation under a rolling window, Refer: Fan and Yao (2003) or Corsi, et al. (2010)
    :param df: logreturn sample
    :param ii: index to search corresponding sample
    :param cv: coefficient of threshold parameter
    :return: local variance estimation of a given period
    """
    # print(f'{ii[0]}-{ii[-1]}')
    R = df.loc[df['ii'].isin(ii), 'log_return'].values  # Get corresponding logreturn from logreturn df
    V_pre = df.loc[df['ii'].isin(ii), 'V_est'].values  # Get previous variance estimation, init with np.inf
    # x = np.array(range(len(R))) / len(R)  # Half gaussian kernel, for test
    x = np.linspace(start=-1, stop=1, num=len(ii))  # x for gaussian kernel K(x)

    # Gaussian kernel on i/L and indicator function, for every observation
    gaussian_kernel = np.exp((-x ** 2) / 2) / (np.sqrt(2 * np.pi))
    # index function, 1 when r^2 <= cv*V_pre, 0: else, for every observation
    idx_func = list(map(lambda r, v: 1 if np.square(r) <= (cv ** 2) * v else 0, R, V_pre))

    # Drop the i=-1,0,1 elements, i = [-L,L]
    gaussian_kernel, R, idx_func = list(map(lambda copo:
                                            np.delete(
                                                copo, [len(copo) // 2, len(copo) // 2 - 1, len(copo) // 2 + 1]
                                            ),
                                            (gaussian_kernel, R, idx_func)))

    denominator = sum(gaussian_kernel * np.square(R) * idx_func)
    numerator = sum(gaussian_kernel * idx_func)
    V_estimate = denominator / numerator
    # print(f'Estimated Variance: {V_estimate[0]}')
    return V_estimate


def variation_estimation_single_period(daily_sample, iterate_num, cv):
    print(daily_sample.head(1).index[0])
    # daily_sample.loc[:, 'V_est'] = np.inf
    daily_sample['V_est'] = np.inf
    daily_sample['ii'] = range(len(daily_sample))
    # daily_sample.loc[:, 'ii'] = range(len(daily_sample))
    for Z in range(iterate_num):
        print(Z)
        # print(daily_sample)
        print(f'{Z} iterations')
        V_est = daily_sample['ii'].rolling(window=window_size, center=True).apply(lambda x: local_variation_estimate(
            df=daily_sample,
            ii=x,
            cv=cv))
        daily_sample.loc[V_est.index, 'V_est'] = V_est.values

        if Z >= 1:
            # print(V_pre)
            # print(V_est)
            V_diff = max(abs(V_pre - V_est.dropna()))
            print(f'Difference: {V_diff}')
            if V_diff < 0.0001:
                print('Converged!')
                # print(daily_sample['V_est'].dropna())
                return daily_sample['V_est']
            print(f'Difference: {V_diff}')
        V_pre = daily_sample['V_est'].dropna()
        daily_sample['V_est'].fillna(np.inf, inplace=True)
    print(f'###iteration end, but may not be converged###: {V_diff}')
    # print(daily_sample['V_est'].dropna())
    return daily_sample['V_est']


def variation_estimation_full_period(full_sample, iterate_num, cv):
    full_sample['V_est'] = np.inf
    full_sample['ii'] = range(len(full_sample))
    for Z in range(iterate_num):
        print(Z)
        V_est = full_sample['ii'].rolling(window=window_size, center=True).apply(lambda x: local_variation_estimate(
            df=full_sample,
            ii=x, cv=cv))
        full_sample.loc[V_est.index, 'V_est'] = V_est.values
        if Z >= 1:
            V_diff = max(abs(V_pre - V_est.dropna()))
            print(f'Difference: {V_diff}')
            if V_diff < 0.001:
                print('Converged!')
                return full_sample['V_est']
        V_pre = full_sample['V_est'].dropna()
        full_sample['V_est'].fillna(np.inf, inplace=True)
    return full_sample['V_est']


def All_RVs_Separation(ts_df, cv=3, delta=1 / 288, alpha=0.9999, refresh_est=False,
                       coin='gemini_BTC', freq='5min', truncate_zero=True, annualized=True, tz_lag=0):
    # Check estimation existence:
    dir_name = f'AllRVs_{coin}/'
    os.makedirs(outdata_dir + f'{dir_name}', exist_ok=True)
    esti_exist = os.path.exists(outdata_dir + f'{dir_name}estimation_{freq}_{cv}.csv')
    timezone = dt.timezone(dt.timedelta(hours=tz_lag))
    if esti_exist and not refresh_est:
        ts_v = pd.read_csv(outdata_dir + f'{dir_name}estimation_{freq}_{cv}.csv', index_col=0, parse_dates=True)
        # ts_v = ts_v.tz_convert(tz=timezone)
        # ts_v = ts_v.tz_localize(tz='UTC').tz_convert(tz=timezone)
        print("===Estimation Existence Check!===")
    else:
        num_iteration_lve = 20
        if isinstance(ts_df, pd.Series):
            ts_df = ts_df.to_frame()

        ts_df.columns = ['log_return']
        ts_df.sort_index(ascending=True, inplace=True)
        ts_df = ts_df[~(ts_df.index.date == ts_df.index.date[0])]
        ts_df = ts_df[~(ts_df.index.date == ts_df.index.date[-1])]
        # for test
        # daily_sample = ts_df[ts_df.index.date == dt.date(2017,7,18)]

        """
        For the following 2 lines:
        # First line estimate within whole sample that might cause using future information problem
        # Second line estimate within each day that doesn't affect RV for next day
        """

        # ts_vest = variation_estimation_full_period(return_df=ts_df, iterate_num=num_iteration_lve, cv=cv)

        # ====test local variation estimation
        # x = ts_df[ts_df.index.date == dt.date(2017, 7, 20)]
        # x_vest = variation_estimation_single_period(x, num_iteration_lve, cv)
        # x_vest.plot()
        # ts_df = ts_df[(ts_df.index.date > dt.date(2017, 1, 1)) & (ts_df.index.date < dt.date(2017, 8, 1))]
        # ====end test

        ts_vest = pd.DataFrame()
        ts_df_grouped = ts_df.groupby(by=ts_df.index.date, axis=0)

        for num, ts_df_daily in ts_df_grouped:
            # # break for test
            # if num == dt.date(2017,7,25):
            #     break
            v_est_daily = variation_estimation_single_period(daily_sample=ts_df_daily,
                                                             iterate_num=num_iteration_lve,
                                                             cv=cv)

            ts_vest = pd.concat([ts_vest, v_est_daily], axis=0)

        # ts_vest = ts_df.groupby(by=ts_df.index.date, axis=0).apply(
        #     lambda x: variation_estimation_single_period(x, num_iteration_lve, cv))

        ts_df['V_est'] = ts_vest
        ts_v = ts_df.dropna()
        # plt.plot(ts_v['V_est'])
        ts_v['theta'] = ts_v['V_est'] * (cv ** 2)
        ts_v['indicate'] = [1 if r ** 2 <= theta else 0 for r, theta in
                            zip(ts_v['log_return'].values, ts_v['theta'].values)]
        # ts_v['indicate'].value_counts() #  1: 212556, 0: 4723
        gammas = [1, 4 / 3, 2]
        for gamma in gammas:
            ts_v[f'z_{round(gamma, 1)}'] = [abs(r) ** (gamma) if ind == 1 else high_vol_smoother(theta, cv, gamma) for
                                            r, ind, theta in zip(
                    ts_v['log_return'].values, ts_v['indicate'].values, ts_v['theta'].values)]
        ts_v.to_csv(outdata_dir + f'{dir_name}estimation_{freq}_{cv}.csv')  # save after estimation immediately

    # plt.figure(figsize=(15, 6))
    # plt.plot(ts_v['z'])
    # plt.plot(ts_v['log_return'])

    # ctrv = ts_v.groupby(by=ts_v.index.date, axis=0).apply(lambda x: (x['z_1'] ** 2).sum(axis=0))

    # =====Calculate C-Threshold Bipower variation
    mu_1 = mu_function(1)
    ts_v['bp'] = ts_v['z_1'].shift(1) * ts_v['z_1']
    ctbpv = ts_v.groupby(by=ts_v.index.date, axis=0).apply(lambda x: x['bp'].sum(axis=0))
    ctbpv = ctbpv / (mu_1 ** 2)

    # =====Calculate C-Threshold Tripower quartacity variation
    mu_43 = mu_function(4 / 3)
    ts_v['trip'] = ts_v['z_1.3'].shift(2) * ts_v['z_1.3'].shift(1) * ts_v['z_1.3']
    cttpv = ts_v.groupby(by=ts_v.index.date, axis=0).apply(lambda x: x['trip'].sum(axis=0))
    cttpv = cttpv / ((mu_43 ** (3)) * delta)

    # concat C-Threshold variations
    trvs = pd.concat([ctbpv, cttpv], axis=1)
    trvs.columns = ['CTBPV', 'CTTPV']

    # annualizing variance by multiplying 365
    if annualized:
        trvs = trvs * 365

    # =====Calculate Realized variation
    # print(ts_df.columns)
    rvs = realized_volatility_separation(return_df=ts_df['log_return'], delta=delta, alpha=alpha,
                                         truncate_zero=truncate_zero, annualized=annualized)

    # concat all variation estimators
    all_RVs = pd.concat([rvs, trvs], axis=1, sort=True)

    # ======Calculate jump test statistics, called ctz
    all_RVs['ctz'] = list(
        map(lambda rv, ctbpv, cttpv: jump_test_statistics(rv, ctbpv, cttpv), all_RVs['RV'], all_RVs['CTBPV'],
            all_RVs['CTTPV']))
    all_RVs['ctz'] = all_RVs['ctz'] / np.sqrt(delta)
    # all_RVs['ctz'].plot()

    # ====Calculate CT-Jump
    ct_jump = all_RVs['RV'] - all_RVs['CTBPV']
    if truncate_zero:
        ct_jump[ct_jump < 0] = 0
    all_RVs['CTJump_raw'] = ct_jump.copy(deep=True)
    sig_level = norm.ppf(alpha)
    sig_ind = [1 if z_value > sig_level else 0 for z_value in all_RVs['ctz']]
    all_RVs[f'CTJump_{alpha}'] = (ct_jump) * sig_ind

    return all_RVs, ts_v


def realized_volatility(self):
    _rvs = realized_volatility_separation(return_df=self.logreturn_hf)
    self.rv = _rvs[0]
    self.bpv = _rvs[1]
    self.jump = _rvs[2]


def CAL_AllRVs(coin, freq, sample_num, cv, refresh, refresh_est, truncate_zero, alpha, annualized, tz_lag):
    # timezone = dt.timezone(dt.timedelta(hours=tz_lag))
    var_check = ['allrvs', 'estimation']
    # delta = 1 / (1440 / int(freq.split('min')[0]))
    delta = 1 / sample_num
    log_return = read_coin_return(coin, freq, refresh, tz_lag)  # tz_lag is not functioning

    dir_name = f'AllRVs_{coin}/'
    os.makedirs(outdata_dir + f'{dir_name}', exist_ok=True)

    if_exist = [os.path.exists(outdata_dir + f'{dir_name}{var}_{freq}_{cv}.csv') for var in var_check]

    if all(if_exist) and not refresh:
        all_RVs = pd.read_csv(outdata_dir + f'{dir_name}allrvs_{freq}_{cv}_{alpha}.csv', index_col=0, parse_dates=True)
        estimation = pd.read_csv(outdata_dir + f'{dir_name}estimation_{freq}_{cv}.csv', index_col=0, parse_dates=True)
        # TODO: adaptive to different timezone
        # all_RVs = all_RVs.tz_localize('UTC').tz_convert(timezone)
        # estimation = estimation.tz_localize('UTC').tz_convert(timezone)
        print("===Existence Check!===")
    else:
        all_RVs, estimation = All_RVs_Separation(ts_df=log_return,
                                                 cv=cv,
                                                 delta=delta,
                                                 alpha=alpha,
                                                 refresh_est=refresh_est,
                                                 coin=coin,
                                                 freq=freq,
                                                 truncate_zero=truncate_zero,
                                                 annualized=annualized)

        all_RVs.to_csv(outdata_dir + f'{dir_name}allrvs_{freq}_{cv}_{alpha}.csv')
    all_RVs.dropna(axis=0, inplace=True)

    temp_truncate_date = dt.date(2018,9,1)

    # truncate the data for now
    all_RVs = all_RVs[all_RVs.index <= temp_truncate_date]
    log_return = log_return[log_return.index.date <= temp_truncate_date]
    estimation = estimation[estimation.index.date <= temp_truncate_date]

    return all_RVs, estimation, log_return


def main_test_func():
    # ======Test Code for class object
    # coins = ['gemini_BTC', 'gemini_ETH']
    #
    # freqs = ['5min']
    # cvs = [3]
    # alpha = 0.9999
    # sample_num = 288
    # tz_lag = 0

    # For test special case
    # logreturn_hf, default = read_coin_diff_freq(coin)
    # coin = coins[0]
    # freq = freqs[0]
    # cv = cvs[0]
    # alpha = alphas[0]

    params = itertools.product(coins, freqs,alphas, cvs)
    for coin, freq,alpha, cv in params:
        print(f'{coin} at {freq}, cv={cv}, alpha={alpha}')
        try:
            # ! REFRESH ALL ESTIMATION RESULTS
            all_RVs, estimation, logreturn = CAL_AllRVs(coin=coin,
                                                        freq=freq,
                                                        cv=cv,
                                                        refresh=True,
                                                        sample_num=sample_num,
                                                        refresh_est=True,
                                                        truncate_zero=True,
                                                        alpha=alpha,
                                                        annualized=True,
                                                        tz_lag=tz_lag)
        except Exception as e:
            print(e)
            continue




"""
PLT_RV_BPV.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import pickle
import itertools

mpl.use('TKAgg')
import matplotlib.pyplot as plt
import datetime as dt
from Code.RCVJ_RealizedVolatility.CAL_RV_BPV import All_RVs_Separation
from Code.RCVJ_RealizedVolatility.CAL_RV_BPV import CAL_AllRVs
from Code.RCVJ_RealizedVolatility.PLT_Funcs import *
from Code.RCVJ_DataOperation.IO_Funcs import read_coin_return, read_indices_full_RV, read_indices_trading_rv_bpv
from statsmodels.nonparametric.kde import KDEUnivariate as uni_kde
from Code.RCVJ_DataOperation.IO_Funcs import read_dyos_clean, read_clean_gemini
from Code.GlobalParams import *


def plt_local_variation_estimate(threshold_rvs, cv, logreturn):
    # ====== plot local variation estimate
    # try:
    #     threshold_rvs
    # except NameError:
    #     threshold_rvs = All_RVs_Separation(ts_df=logreturn, cv=cv)

    ts_v = threshold_rvs[1]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(ts_v['V_est'])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=14)
    ax.set_title('Local Variation Estimate')
    fig.savefig(outplot_dir + 'local_variation_estimate.png', dpi=300)

    # def plt_trvs_with_diffcvs(cvs):
    #     # =========plot threshold realized jumps with different cvs
    #     for cv in cvs:
    #         plt_threshold_realized_jumps(cv)


def plt_highlow_rvs_intraday(logreturn):
    # =======
    # rv, bpv, jump = rvs
    # rv_crypto_sort = rv.sort_values(by='RV', axis=0, ascending=False)
    # bpv_crypto_sort = bpv.sort_values(by='BPV', axis=0, ascending=False)

    high_vol_date = dt.date(2018, 1, 16)
    low_vol_date = dt.date(2016, 10, 6)

    t1 = logreturn[logreturn.index.date == high_vol_date]
    t2 = logreturn[logreturn.index.date == low_vol_date]

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t1, color='blue')
    plt.ylim(-0.08, 0.03)
    plt.ylabel('Log Return', fontsize=16)
    plt.title(f"Return on High Volatility Day ({high_vol_date.strftime('%Y-%m-%d')})")

    plt.subplot(2, 1, 2)
    plt.plot(t2, color='blue')
    plt.title(f"Return on Low Volatility Day ({low_vol_date.strftime('%Y-%m-%d')})")
    plt.ylim(-0.08, 0.03)
    plt.ylabel('Log Return', fontsize=16)
    plt.tight_layout()

    plt.savefig('high_low_volatility.png', dpi=300)


def plt_kernel_estimation(ts=None, bin_num=1000, bandwidth=None, bd_method='silverman', hist_density=True,
                          var_name='RV', cv=3, asset_name='gemini_BTC', freq='5min'):
    # ====================== Plot kernel estimation
    # plot kernel estimation
    kde_dir = outplot_dir + f'KDE/KDE_{asset_name}/'
    os.makedirs(kde_dir, exist_ok=True)
    kernel_plotter = PlotKernelDensityEstimator(data_points=ts.values)
    kde_fig = kernel_plotter.plot_curve_hist_kde(bin_num=bin_num,
                                                 hist_density=hist_density,
                                                 bandwidth=bandwidth,
                                                 method=bd_method)
    op_bd = round(kernel_plotter.bandwidth, 2)
    kde_file_name = kde_dir + f'{var_name}_{kernel_plotter.kernel}_{op_bd}_cv{cv}_{freq}.png'
    kde_fig.savefig(kde_file_name, dpi=300)


def plt_logretrun_distribution(logreturn, asset_name, freq, drop_zero=False, rand_sample_size=None, replace=False):
    # ====================== Plot distribution of log returns

    # coin_name = 'BTC'
    # freq = '5min'
    # logreturn = read_dyos_clean(coin_name)

    logreturn = logreturn[logreturn.index.date >= dt.date(2017, 1, 1)]

    logreturn_drop_zero = logreturn[~(logreturn[f'{asset_name}_5min'] == 0)]

    value_count = logreturn_drop_zero[f'{asset_name}_5min'].value_counts()

    # =========Test for
    high_counts = value_count.head(1).index[0]
    high_counts_day = logreturn_drop_zero[logreturn_drop_zero['BTC_5min'] == high_counts]
    high_counts_day['day'] = high_counts_day.index.date

    # counts_day = high_counts_day['day'].value_counts()
    # close_price = pd.read_csv(data_dir + f'/{asset_name}_USDT.csv', index_col=0, parse_dates=True)['close']
    # volume = pd.read_csv(data_dir + f'/{asset_name}_USDT.csv', index_col=0, parse_dates=True)['volume']
    # high_days = counts_day.head(20).index
    # for high_day in high_days:
    #     high_day_price = close_price[close_price.index.date == high_day]
    #     plt.figure(figsize=(15, 5))
    #     plt.plot(high_day_price)
    #     plt.title(f'Intraday Close Price, {high_day}')
    #     plt.savefig(outplot_dir + f"HighFreqSameReturn/OneDaySample_{high_day}.png", dpi=300)

    top_values = value_count[value_count > 1]
    values_todrop = top_values.index
    logreturn_dropped = logreturn_drop_zero[~(logreturn_drop_zero[f'{asset_name}_5min'].isin(values_todrop))]
    data_df = logreturn_dropped
    # value_count_after = logreturn_dropped['BTC_5min'].value_counts()

    histogram = plot_logreturn_hist(data_df=data_df,
                                    bins=20000,
                                    hist_density=False,
                                    drop_zero=drop_zero,
                                    random_sample_size=rand_sample_size,
                                    replace=replace,
                                    xlabel='Log-return')

    # kernel_plotter = PlotKernelDensityEstimator(data_points=data_df.values,
    #                                             file_name=f'KDE/KDE_{asset_name}_{var_name}')
    # kernel_plotter.band_widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # kernel_plotter.x_plot = np.linspace(data_df.min(), data_df.max(), 100000)
    # kernel_plotter.log_density_estimate()
    # kernel_plotter.plot_curve_hist_kde(log_dens=kernel_plotter.log_densities['0.1'], bin_num=20000, hist_density=True)
    # kernel_plotter.save_plot(dir=outplot_dir, num_bin=40000, hist_density=True)

    # histogram.savefig(outplot_dir + f'{asset_name}_{freq}_logreturn_distribution.png', dpi=300)
    histogram[0].savefig(outplot_dir + f'{asset_name}_{freq}_logreturn_distribution_2017.png', dpi=500)


def plot_program_trading(asset_name='BTC'):
    """
    In datset BTC-D, we will find exact same log return within some of the trading days
    which leads us to understand as evidence of program trading
    :param logreturn:
    :param asset_name:
    :return:
    """

    logreturn, _default = read_dyos_clean(asset_name)
    close_price = _default['close']
    volume = _default['volume']

    logreturn_drop_zero = logreturn[~(logreturn['log_return'] == 0)]

    value_count = logreturn_drop_zero['log_return'].value_counts()

    high_counts = value_count.head(1).index[0]
    high_counts_day = logreturn_drop_zero[logreturn_drop_zero['log_return'] == high_counts]
    high_counts_day['day'] = high_counts_day.index.date
    counts_day = high_counts_day['day'].value_counts()
    high_days = counts_day.head(20).index

    for high_day in high_days:
        high_day_price = close_price[close_price.index.date == high_day]
        return_daily = logreturn[logreturn.index.date == high_day]
        return_daily.columns = ['LogReturn']
        return_daily = return_daily['LogReturn']
        high_day_volume = volume[volume.index.date == high_day]
        price_return_plot = plot_price_return_volume(time_index=high_day_price.index,
                                                     price=high_day_price,
                                                     logreturn=return_daily,
                                                     volume=high_day_volume,
                                                     high_count_value=high_counts)
        out_dir = outplot_dir + 'HighFreqSameReturn/'
        os.makedirs(out_dir, exist_ok=True)
        price_return_plot.savefig(outplot_dir + f"HighFreqSameReturn/{asset_name}_OneDaySample_{high_day}.png", dpi=300)


def plt_threshold_realized_jumps(logreturn, rv, bpv, jump, **kwargs):
    # self.threshold_rvs = threshold_realized_volatility_separation(ts_df=self.logreturn, cv=cv)
    # ctrvs = self.threshold_rvs[0]
    # rv = self.rvs[0]
    # ctrvs['ct_jumps'] = rv.values - ctrvs['C_TBPV'].to_frame().values
    title_rv = '$RV^{1/2}$' if 'title_rv' not in kwargs else kwargs['title_rv']
    title_bpv = '$BPV^{1/2},$' if 'title_bpv' not in kwargs else kwargs['title_bpv']
    title_jump = '$Jump^{1/2}$' if 'title_jump' not in kwargs else kwargs['title_jump']

    trvs_fig = plot_RV_separation(return_df=logreturn,
                                  rv=rv,
                                  bpv=bpv,
                                  jump=jump,
                                  title_rv=title_rv,
                                  title_bpv=title_bpv,
                                  title_jump=title_jump
                                  )

    # trvs_fig.savefig(outplot_dir + f'RV_Separation/{coin_name}_{self.freq}_{cv}_CThresholdJumps.png', dpi=300)
    return trvs_fig


def plot_crypto_jumpseparation(coin, freq, all_RVs, logreturn, cv=3, alpha=0.9999):
    """

    :param coin:
    :param freq:
    :param all_RVs:
    :param logreturn:
    :param cv:
    :param alpha:
    :return:
    """

    """
    'RV', 'BPV', 'TPV', 'z', 'Jump_raw', 'Jump_sig', 'CTBPV', 'CTTPV', 'ctz', 'CTJump_raw', 'CTJump_sig'
    """

    rv = all_RVs['RV']
    bpv = all_RVs['BPV']
    ctpv = all_RVs['CTBPV']
    jump_raw = all_RVs['Jump_raw']
    jump_sig = all_RVs[f'Jump_{alpha}']
    tjump = all_RVs['CTJump_raw']
    tjump_sig = all_RVs[f'CTJump_{alpha}']

    # plotter = RealizedVolatilityPlot_OneAsset(return_ts=logreturn, rv=rv, bpv=bpv, asset_name=coin, ts_freq=freq)
    rv_plot_dir = outplot_dir + f'RV_Separation/{coin}/'
    os.makedirs(rv_plot_dir, exist_ok=True)

    # Plot threshold RV separation
    # Raw Jump from BPV
    # jump = rv - bpv
    rvs_fig = plt_threshold_realized_jumps(logreturn=logreturn,
                                           rv=rv,
                                           bpv=bpv,
                                           jump=jump_raw,
                                           title_bpv='$BPV^{1/2}$',
                                           title_jump='$J^{1/2}$')

    rvs_fig.savefig(f'{rv_plot_dir}{coin}_{freq}_BPVRawJumps_{cv}.png', dpi=300)
    # plt.close()

    # BPV significant Jumps
    rvs_sig_fig = plt_threshold_realized_jumps(logreturn=logreturn,
                                               rv=rv,
                                               bpv=bpv,
                                               jump=jump_sig,
                                               title_bpv='$BPV^{1/2}$',
                                               title_jump='$J^{1/2},$' + r'$\alpha$' + f"$={alpha}$")

    rvs_sig_fig.savefig(f'{rv_plot_dir}{coin}_{freq}_BPVSignificantJumps_{cv}_{alpha}.png', dpi=300)
    # plt.close()

    # CTBPV raw Jumps
    ctrvs_fig = plt_threshold_realized_jumps(logreturn=logreturn,
                                             rv=rv,
                                             bpv=ctpv,
                                             jump=tjump,
                                             title_bpv='$TBPV^{1/2},$' + r'$C_\theta$' + f"={cv}",
                                             title_jump='$TJ^{1/2}$')

    ctrvs_fig.savefig(f'{rv_plot_dir}{coin}_{freq}_CTRawJumps_{cv}.png', dpi=300)
    # plt.close()

    # CTBPV significant Jumps

    ctrvs_sig_fig = plt_threshold_realized_jumps(logreturn=logreturn,
                                                 rv=rv,
                                                 bpv=ctpv,
                                                 jump=tjump_sig,
                                                 title_bpv='$TBPV^{1/2},$' r'$C_\theta$' + f"={cv}",
                                                 title_jump='$TJ^{1/2},$' + r'$\alpha$' + f"$={alpha}$")

    ctrvs_sig_fig.savefig(f'{rv_plot_dir}{coin}_{freq}_CTBPVSignificantJumps_{cv}_{alpha}.png', dpi=300)
    # plt.close()

    # plt.plot(all_RVs[f'Jump_raw'])


def plot_index_jumpsepa(index):
    print(index)
    indices_plot_dir = outplot_dir + 'indices_rv/'
    os.makedirs(indices_plot_dir, exist_ok=True)

    rv, bpv, jump = read_indices_trading_rv_bpv()
    rv_asset = rv[index]
    bpv_asset = bpv[index]
    jump_asset = jump[index]

    plt.figure(figsize=(15, 7))
    plt.title(index, fontsize=16)
    plt.subplot(3, 1, 1)
    plt.plot(np.sqrt(rv_asset), color='b')
    plt.ylabel("$\sqrt{RV}$", fontsize=16)

    plt.subplot(3, 1, 2)
    plt.plot(np.sqrt(bpv_asset), color='c')
    plt.ylabel('$\sqrt{BPV}$', fontsize=16)

    plt.subplot(3, 1, 3)
    plt.plot(jump_asset, color='k')
    plt.ylabel('RV-BPV', fontsize=16)

    plt.xlabel('Date', fontsize=16)
    plt.tight_layout()
    plt.savefig(indices_plot_dir + f'{index[1:]}.png', dpi=300)
    plt.close()


def plot_extrem_volatile_day(coin, freq, all_RVs, logreturn, alpha):
    # plot daily price, rv, big significant jumps
    sp_date = dt.date(2017, 3, 10)
    coin_data = read_clean_gemini(coin, freq, True)
    tj_sorted = all_RVs[f'CTJump_{alpha}'].sort_values(ascending=False)
    high_tj_dates = tj_sorted.head(10).index

    close = coin_data['Close']
    close_daily = close.groupby(by=close.index.date, axis=0).apply(lambda x: x.tail(1).sum())

    # close['date'] = close.index.date
    # close_hightj = close[close['date'].isin(high_tj_dates)]
    # close_hightj.drop('date',axis=1,inplace=True)
    # close_hightj.plot()

    # ==== Test
    # close_daily.plot()
    # rv_sigjump = all_RVs[['RV', f'CTJump_{alpha}']]
    # plt.figure(figsize=(15, 8))
    # plt.subplot(3, 1, 1)
    # plt.plot(close_daily, color='black')
    # plt.subplot(3, 1, 2)
    # plt.plot(rv_sigjump['RV'], color='blue')
    # ax3 = plt.subplot(3, 1, 3)
    # plt.plot(rv_sigjump[f'CTJump_{alpha}'], color='red')
    # # Add stripe
    # for timestamp in high_tj_dates:
    #     ax3.axvline(x=timestamp, linewidth=2, color='green')

    # diff = all_RVs['CTJump_raw'] - all_RVs['Jump_raw']
    # sort_diff = diff.sort_values(axis=0, ascending=False)
    # coin_data = read_clean_gemini(coin, freq, True)

    logreturn_sp = logreturn[logreturn.index.date == sp_date]
    vol_sp = coin_data[coin_data.index.date == sp_date]['Volume']
    fig = plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    # plt.scatter(x=logreturn_sp.index, y=logreturn_sp.values)
    plt.plot(logreturn_sp, color='r', linestyle='--')
    plt.ylabel('Log-return', fontsize=17)

    plt.subplot(2, 1, 2)
    plt.plot(vol_sp)
    plt.ylabel('Volume', fontsize=17)
    plt.xlabel('Time', fontsize=17)
    plt.tight_layout()
    fig.savefig(outplot_dir + 'extremly_volatile_day.png', dpi=300)


def plot_all_indices_rvjump():
    all_indices = ['.AEX', '.AORD', '.BFX', '.BSESN', '.BVLG', '.BVSP', '.DJI', '.FCHI', '.FTMIB', '.FTSE', '.GDAXI',
                   '.GSPTSE', '.HSI', '.IBEX', '.IXIC', '.KS11', '.KSE', '.MXX', '.N225', '.NSEI', '.OMXC20', '.OMXHPI',
                   '.OMXSPI', '.OSEAX', '.RUT', '.SMSI', '.SPX', '.SSEC', '.SSMI', '.STI', '.STOXX50E']

    for index in all_indices:
        # index = '.FCHI'
        plot_index_jumpsepa(index)


def plot_kde(ts, bin_num, bandwidth, hist_density, var_name, cv, asset_name, freq, bd_method):
    # coin = 'gemini_BTC'
    # freq = '5min'
    # cv = 3
    # alpha = 0.99
    # refresh = True

    # Plot histogram of ts, bpv, ctbpv, c-jump, jump

    # KDE on RV

    ts = ts[~(ts == 0)]
    log_ts = np.log(ts)
    sqrt_ts = np.sqrt(ts)

    if bin_num is None:
        bin_num = int(len(ts) / 10)

    # plotter = RealizedVolatilityPlot_OneAsset(return_ts=logreturn, rv=ts, asset_name=coin, ts_freq=freq)

    """
    ts=None, bin_num=1000, bandwidth=None, bd_method='silverman', hist_density=True,
    var_name='RV', cv=3, asset_name='gemini_BTC', freq='5min'
    """

    plt_kernel_estimation(ts=log_ts, bin_num=bin_num, bandwidth=bandwidth, hist_density=hist_density,
                          var_name=f'Log_{var_name}',
                          cv=cv, asset_name=asset_name, freq=freq)

    plt_kernel_estimation(ts=sqrt_ts, bin_num=bin_num, bandwidth=bandwidth, hist_density=hist_density,
                          var_name=f'Sqrt_{var_name}',
                          cv=cv, asset_name=asset_name, freq=freq)

    plt_kernel_estimation(ts=ts / ts.mean(), bin_num=bin_num, bandwidth=bandwidth, hist_density=hist_density,
                          var_name=f'centered_{var_name}',
                          cv=cv, asset_name=asset_name, freq=freq)

    plt_kernel_estimation(ts=sqrt_ts / sqrt_ts.mean(), bin_num=bin_num, bandwidth=bandwidth, hist_density=hist_density,
                          var_name=f'centered_sqrt_{var_name}',
                          cv=cv, asset_name=asset_name, freq=freq)

    plt_kernel_estimation(ts=log_ts / log_ts.mean(), bin_num=bin_num, bandwidth=bandwidth, hist_density=hist_density,
                          var_name=f'centered_Log_{var_name}',
                          cv=cv, asset_name=asset_name, freq=freq)


def plot_crypto_rvs_kde(all_RVs, coin, freq, cv, **kwargs):
    all_RVs.dropna(axis=0, inplace=True)

    # KDE on BPV
    rv = all_RVs['RV']
    bpv = all_RVs['BPV']
    tbpv = all_RVs['CTBPV']
    # jump_sig = all_RVs['Jump_0.99']
    # tjump_sig = all_RVs['CTJump_0.99']
    """
    ts, bin_num, bandwidth, hist_density, var_name, cv, asset_name, freq, bd_method
    """
    plot_kde(ts=rv, bin_num=kwargs['bin_num'], bandwidth=kwargs['bandwidth'], hist_density=True, var_name='RV',
             asset_name=coin, freq=freq, cv=cv, bd_method='silverman')
    plot_kde(ts=bpv, bin_num=kwargs['bin_num'], bandwidth=kwargs['bandwidth'], hist_density=True, var_name='BPV',
             asset_name=coin, freq=freq, cv=cv, bd_method='silverman')
    plot_kde(ts=tbpv, bin_num=kwargs['bin_num'], bandwidth=kwargs['bandwidth'], hist_density=True, var_name='TBPV',
             asset_name=coin, freq=freq, cv=cv, bd_method='silverman')
    # plot_kde(ts=jump_sig, rv_name='JumpSig', coin=coin, freq=freq, cv=cv)
    # plot_kde(ts=tjump_sig, rv_name='TJump_sig', coin=coin, freq=freq, cv=cv)


def plot_indices_kde():
    all_indices = ['.AEX', '.AORD', '.BFX', '.BSESN', '.BVLG', '.BVSP', '.DJI', '.FCHI', '.FTMIB', '.FTSE', '.GDAXI',
                   '.GSPTSE', '.HSI', '.IBEX', '.IXIC', '.KS11', '.KSE', '.MXX', '.N225', '.NSEI', '.OMXC20', '.OMXHPI',
                   '.OMXSPI', '.OSEAX', '.RUT', '.SMSI', '.SPX', '.SSEC', '.SSMI', '.STI', '.STOXX50E']

    rv = read_indices_full_RV()

    for index in all_indices:
        print(index)
        rv_index = rv[index]
        plot_kde(ts=rv_index, bin_num=100, bandwidth=0.5, hist_density=True, var_name='RV', asset_name=index, cv=0,
                 freq='5min', bd_method='silverman')


def plot_funcs(**kwargs):
    # logreturn_hf, default = read_coin_diff_freq(coin)
    # coin = coins[0]
    # freq = freqs[0]
    # cv = cvs[0]

    params = itertools.product(coins, freqs, cvs)

    for coin, freq, cv in params:

        print(f'{coin} in {freq}, cv={cv}')

        all_RVs, estimation, logreturn = CAL_AllRVs(coin,
                                                    freq,
                                                    refresh=True,
                                                    refresh_est=False,
                                                    sample_num=sample_num,
                                                    cv=cv,
                                                    truncate_zero=True,
                                                    alpha=alpha,
                                                    annualized=True,
                                                    tz_lag=tz_lag)

        if 'rvs_jump' in kwargs:
            # Plot RV, BPV and jump separation on specified Cryptos
            plot_crypto_jumpseparation(coin=coin, freq=freq, all_RVs=all_RVs, logreturn=logreturn, cv=cv, alpha=alpha)

        if 'kde' in kwargs:
            # Plot KDE
            plot_crypto_rvs_kde(all_RVs=all_RVs, logreturn=logreturn, coin=coin, freq=freq, cv=cv, bin_num=100, bandwidth=1.5)

        if 'kde_logreturn' in kwargs:
            # Plot KDE on logreturn
            plt_logretrun_distribution(logreturn=logreturn, asset_name=coin, freq=freq, drop_zero=True)

        if 'program_trading' in kwargs:
            plot_program_trading()

```

automatically created on 2019-04-04