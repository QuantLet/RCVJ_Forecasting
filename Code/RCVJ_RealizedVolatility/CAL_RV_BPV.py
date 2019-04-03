import matplotlib as mpl

mpl.use('TKAgg')
from Code.RCVJ_DataOperation.IO_Funcs import read_coin_return
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
    idxs = np.array(range(len(R))) / len(R)

    # Gaussian kernel on i/L and indicator function
    gaussian_kernel = np.exp((-idxs ** 2) / 2) / (np.sqrt(2 * np.pi))
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
    daily_sample.loc[:, 'V_est'] = np.inf
    daily_sample.loc[:, 'ii'] = range(len(daily_sample))
    for Z in range(iterate_num):
        print(Z)
        # print(daily_sample)
        print(f'{Z} iterations')
        V_est = daily_sample['ii'].rolling(window=51, center=True).apply(lambda x: local_variation_estimate(
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
        V_est = full_sample['ii'].rolling(window=51, center=True).apply(lambda x: local_variation_estimate(
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
            # if num == dt.date(2018,8,23):
            #     break
            v_est_daily = variation_estimation_single_period(daily_sample=ts_df_daily, iterate_num=num_iteration_lve,
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

    params = itertools.product(coins, freqs,alphas, cvs)
    for coin, freq,alpha, cv in params:
        print(f'{coin} at {freq}, cv={cv}, alpha={alpha}')
        try:
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
