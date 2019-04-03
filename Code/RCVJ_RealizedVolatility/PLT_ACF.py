from Code.GlobalParams import *
import matplotlib as mlp
mlp.use('TKAgg')

import os
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
from Code.RCVJ_DataOperation.IO_Funcs import read_indices_full_RV
import itertools
import pandas as pd
from Code.RCVJ_RealizedVolatility.CAL_RV_BPV import CAL_AllRVs
from Code.RCVJ_RealizedVolatility.PLT_Funcs import plot_acf


def multiple_acfs(var_ts, max_lags, alpha):
    """

    :param var_ts:
    :param max_lags:
    :param alpha:
    :return:
    """

    # Compute ACF coeffiecnts
    acfs = pd.DataFrame(columns=var_ts.columns)
    for col in var_ts.columns:
        print(col)
        ts_uni = var_ts[col]
        ts_uni.dropna(inplace=True)
        acc, cofint = acf(ts_uni, nlags=max_lags, alpha=alpha)
        acfs.loc[:, col] = acc
        # acfs.plot()
    return acfs


def construct_variables_over_cryptos(freq, alpha, cv, annualized=True, ):
    """
    Construct each variable over different asset
    :return:
    """
    stats_out_dir = outdata_dir + f'Vars_AllCryptos/'
    os.makedirs(stats_out_dir, exist_ok=True)

    assets = ['gemini_BTC', 'gemini_ETH', 'BTC', 'ETH', 'XRP', 'LTC']

    # all params
    refresh = True
    truncate_zero = True
    # annualized = True

    #
    # freq = freqs[0]
    # alpha = alphas[0]
    # cv = cvs[1]

    # Extract RV variable over all cryptos
    rvs_file = outdata_dir + f'Vars_AllCryptos/RV_{freq}_{alpha}_{cv}.csv'
    zs_file = outdata_dir + f'Vars_AllCryptos/z-stat_{freq}_{alpha}_{cv}.csv'
    tzs_file = outdata_dir + f'Vars_AllCryptos/tz-sta_{freq}_{alpha}_{cv}.csv'
    js_file = outdata_dir + f'Vars_AllCryptos/Jump_Sig_{freq}_{alpha}_{cv}.csv'
    tjs_file = outdata_dir + f'Vars_AllCryptos/tJump_sig_{freq}_{alpha}_{cv}.csv'

    exist_flag = all([os.path.exists(file_name) for file_name in [rvs_file, zs_file, tzs_file, js_file, tjs_file]])

    if exist_flag:
        rvs = pd.read_csv(rvs_file, index_col=0, parse_dates=True)
        zs = pd.read_csv(zs_file, index_col=0, parse_dates=True)
        tzs = pd.read_csv(tzs_file, index_col=0, parse_dates=True)
        js = pd.read_csv(js_file, index_col=0, parse_dates=True)
        tjs = pd.read_csv(tjs_file, index_col=0, parse_dates=True)

    else:

        rvs = pd.DataFrame(columns=assets)
        zs = pd.DataFrame(columns=assets)
        tzs = pd.DataFrame(columns=assets)
        js = pd.DataFrame(columns=assets)
        tjs = pd.DataFrame(columns=assets)

        for coin in rvs.keys():
            print(coin)
            all_RVs, estimation, logreturn = CAL_AllRVs(coin=coin,
                                                        freq=freq,
                                                        refresh=refresh,
                                                        refresh_est=False,
                                                        cv=cv,
                                                        truncate_zero=truncate_zero,
                                                        alpha=alpha,
                                                        annualized=annualized,
                                                        tz_lag=0)
            rvs.loc[:, coin] = all_RVs['RV']
            zs.loc[:, coin] = all_RVs['z']
            tzs.loc[:, coin] = all_RVs['ctz']
            js.loc[:, coin] = all_RVs[f'Jump_{alpha}']
            tjs.loc[:, coin] = all_RVs[f'CTJump_{alpha}']

        rvs.to_csv(outdata_dir + f'Vars_AllCryptos/RV_{freq}_{alpha}_{cv}.csv')
        zs.to_csv(outdata_dir + f'Vars_AllCryptos/z-stat_{freq}_{alpha}_{cv}.csv')
        tzs.to_csv(outdata_dir + f'Vars_AllCryptos/tz-sta_{freq}_{alpha}_{cv}.csv')
        js.to_csv(outdata_dir + f'Vars_AllCryptos/Jump_Sig_{freq}_{alpha}_{cv}.csv')
        tjs.to_csv(outdata_dir + f'Vars_AllCryptos/tJump_sig_{freq}_{alpha}_{cv}.csv')

    log_rvs = rvs.replace(0, np.nan)
    log_rvs = np.log(log_rvs)

    log_js = np.log(js + 1)
    log_tjs = np.log(tjs + 1)

    cs = rvs - js
    log_cs = cs.replace(0, np.nan)
    log_cs = np.log(log_cs)

    tcs = rvs - tjs
    log_tcs = tcs.replace(0, np.nan)
    log_tcs = np.log(log_tcs)

    return rvs, log_rvs, js, log_js, tjs, log_tjs, cs, log_cs, tcs, log_tcs



def acf_plot_over_cryptos(acf_coef, file_name):
    # acf_coef = acfs

    # Plot ACF coefficient decay over all cryptos
    os.makedirs(outplot_dir + f'acf_plots/', exist_ok=True)

    colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    styles = ['-', '--', ':', '-.', ':', '-', '--']
    labels = ['BTC-G', 'ETH-G', 'BTC-D', 'ETH-D', 'XRP-D', 'LTC-D']
    lines = list(zip(colors, styles, labels))

    fig = plt.figure(figsize=(15, 7))
    for num, asset in enumerate(acf_coef.columns):
        print(num)
        print(asset)
        print(lines[num])
        line_def = lines[num]
        plt.plot(acf_coef[asset], color=line_def[0], linestyle=line_def[1], label=line_def[2])

    plt.xlabel('Lags', fontsize=17)
    plt.ylabel('Autocorrelation', fontsize=17)
    plt.legend()
    plt.xlim([0, len(acf_coef)])
    plt.tight_layout()
    fig.savefig(outplot_dir + f'acf_plots/{file_name}.png', dpi=300)
    print('Plot Finished')


def acf_plot_over_assets(acf_coefs, lines, file_name, max_lags=None):
    os.makedirs(outplot_dir + f'acf_plots/', exist_ok=True)

    if max_lags is not None:
        acf_coefs = acf_coefs.iloc[0:max_lags + 1, :]

    # lines = list(zip(colors, styles, labels))
    fig = plt.figure(figsize=(15, 7))
    for num, asset in enumerate(acf_coefs.columns):
        print(num)
        print(asset)
        line_def = lines[num]
        plt.plot(acf_coefs[asset], color=line_def[0], linestyle=line_def[1], label=asset)

    plt.xlabel('Lags', fontsize=17)
    plt.ylabel('Autocorrelation', fontsize=17)
    plt.legend()
    plt.xlim([0, len(acf_coefs)])
    plt.tight_layout()
    fig.savefig(outplot_dir + f'acf_plots/{file_name}.png', dpi=300)
    print('Plot Finished')


def lines_generator(number_of_lines):
    colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
    styles = ['-', '--', ':', '-.']
    lines = list(itertools.product(colors, styles))
    return lines[0:number_of_lines]


def calculate_cryptos_acf(freq='5min', alpha=0.9999, cv=3):
    rvs, log_rvs, js, log_js, tjs, log_tjs, cs, log_cs, tcs, log_tcs = construct_variables_over_cryptos(freq,
                                                                                                        alpha,
                                                                                                        cv,
                                                                                                        annualized=True)

    log_rv_acf = multiple_acfs(log_rvs, max_lags=365, alpha=0.05)
    log_cs_acf = multiple_acfs(log_cs, max_lags=365, alpha=0.05)
    log_tcs_acf = multiple_acfs(log_tcs, max_lags=365, alpha=0.05)
    return log_rv_acf, log_cs_acf, log_tcs_acf


def plot_cryptos_acf(freq='5min', alpha=0.9999, cv=3):
    crypto_names = ['BTC-G', 'ETH-G', 'BTC-D', 'ETH-D', 'XRP-D', 'LTC-D']
    lines = lines_generator(len(crypto_names))

    log_rv_acf, log_cs_acf, log_tcs_acf = calculate_cryptos_acf(freq, alpha, cv)

    acf_plot_over_assets(log_rv_acf, lines, file_name=f'LogRV_{freq}_{alpha}_{cv}', max_lags=300)
    acf_plot_over_assets(log_cs_acf, lines, file_name=f'LogCs_{freq}_{alpha}_{cv}', max_lags=300)
    acf_plot_over_assets(log_tcs_acf, lines, file_name=f'LogTCs_{freq}_{alpha}_{cv}', max_lags=300)


def plot_indices_acf_compare_btcg(coin='gemini_BTC'):
    # Read Indices RV data
    indices_rv = read_indices_full_RV()
    symbols = ['.AEX', '.DJI', '.FTSE', '.HSI', '.SPX', '.SSEC']
    indices_rv = indices_rv.loc[:, symbols]
    # indices_rv_acf = multiple_acfs(var_ts=indices_rv, max_lags=365, alpha=0.05)
    indices_logrv_acf = multiple_acfs(var_ts=np.log(indices_rv), max_lags=365, alpha=0.05)

    # Read Cryptos RV data
    log_rv_acf, log_cs_acf, log_tcs_acf = calculate_cryptos_acf(freq='5min', alpha=0.9999, cv=3)

    indices = indices_rv.columns

    lines = lines_generator(len(indices) + 1)
    logrv_comp_acf = pd.concat([indices_logrv_acf, log_rv_acf[coin]], axis=1)

    # acf_plot_over_assets(acf_coefs=indices_rv_acf, lines=lines, file_name='Indices_RV_ACF', max_lags=300)
    acf_plot_over_assets(acf_coefs=logrv_comp_acf, lines=lines, file_name='Indices_compare_LogRV_ACF', max_lags=300)

freq = freqs[0]
cv = cvs[0]
coin = coins[0]

plot_cryptos_acf(freq=freq, alpha=alpha, cv=cv)
plot_indices_acf_compare_btcg(coin='gemini_BTC')
plot_indices_acf_compare_btcg(coin='gemini_ETH')
