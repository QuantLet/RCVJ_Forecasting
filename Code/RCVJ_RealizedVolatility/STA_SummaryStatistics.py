import pandas as pd
import statsmodels.tsa as sta
import numpy as np
from Code.RCVJ_RealizedVolatility.CAL_RV_BPV import CAL_AllRVs
from Code.RCVJ_RealizedVolatility.PLT_Funcs import *
from scipy.stats import skew, kurtosis
from Code.RCVJ_DataOperation.IO_Funcs import read_indices_full_RV
from statsmodels.tsa.stattools import acf, pacf
import itertools
from Code.GlobalParams import *


def reformat_stats_number(x, decimal_round, format_threshold):
    # decimal_round=5
    format_string = '{:.' + f'{decimal_round}' + 'f}%'
    return list(
        map(lambda s: format_string.format(s * 100) if abs(s) < format_threshold else np.round(s, decimal_round),
            x.astype(float)))


# return format_string.format(x*100) if x < format_threshold else x


def SummaryStatistics(estimator_matrix, **kwargs):
    """
    describe auxi_stats: count, mean, std, min, percentiles, max
    skewness, kurtosis, Auto-correlation
    :param estimator_matrix:
    :param output_dir:
    :param file_name:
    :param kwargs:
    :return:
    """

    if not isinstance(estimator_matrix, pd.DataFrame):
        estimator_matrix = estimator_matrix.to_frame()

    if 'drop_zero_col' in kwargs:
        drop_zero_col = kwargs['drop_zero_col']
        for col in drop_zero_col:
            estimator_matrix.loc[estimator_matrix[col] == 0, col] = np.nan

    if 'lower_percentile' in kwargs and 'higher_percentile':
        describe_stats = estimator_matrix.describe(percentiles=[kwargs['lower_percentile'], kwargs['higher_percentile']])
    else:
        describe_stats = estimator_matrix.describe()

    variables = estimator_matrix.columns
    skewness = pd.Series(skew(estimator_matrix, axis=0, nan_policy='omit'), variables)
    kurtosises = pd.Series(kurtosis(estimator_matrix, axis=0, nan_policy='omit'), variables)

    auxi_stats = pd.concat([skewness, kurtosises], axis=1)
    auxi_stats.columns = ['skewness', 'kurtosis']

    estimator_matrix.fillna(0, inplace=True)

    if 'acf_lags' in kwargs:
        acf_values = dict()
        for acf_lag in kwargs['acf_lags']:
            acf_vars = list()
            for var in variables:
                print(var)
                acfs, confint = acf(estimator_matrix[var], nlags=365, alpha=0.05)
                acf_vars.append(acfs[acf_lag])
            acf_values[f'acf{acf_lag}'] = acf_vars

        acf_matrix = pd.DataFrame(acf_values, index=variables)
        auxi_stats = pd.concat([auxi_stats, acf_matrix], axis=1)

    summary_stats = pd.concat([describe_stats, np.transpose(auxi_stats)], axis=0, sort=False)

    return summary_stats


def crypto_summarystats(coin='gemini_BTC',
                        freq='5min',
                        cv=3,
                        refresh=True,
                        alpha=0.9999,
                        decimal=2,
                        sample_num=288,
                        format_threshold=0.0001,
                        annualized=True,
                        truncate_zero=True):
    """

    :param coin:
    :param freq:
    :param cv:
    :param refresh:
    :param backup_data:
    :param alpha:
    :return:
    """
    # coin = 'gemini_BTC'
    # freq = '5min'
    # cv = 3
    # refresh = True
    # alpha = 0.9999
    # annualized=True
    # truncate_zero=True
    # decimal = 2

    """
    'RV', 'BPV', 'TPV', 'z', 'Jump_raw', 'Jump_0.99', 'CTBPV', 'CTTPV',
       'ctz', 'CTJump_raw', 'CTJump_0.99'
    """
    # Save dir
    stat_out_dir = outdata_dir + f'SummaryStats/{coin}_{freq}/'
    os.makedirs(stat_out_dir, exist_ok=True)

    # Calculate RVs estimators
    all_RVs, estimation, logreturn = CAL_AllRVs(coin=coin,
                                                freq=freq,
                                                refresh=refresh,
                                                refresh_est=False,
                                                cv=cv,
                                                sample_num=sample_num,
                                                truncate_zero=truncate_zero,
                                                alpha=alpha,
                                                annualized=annualized,
                                                tz_lag=0)

    # Preprocessing
    all_RVs.dropna(axis=0, inplace=True)
    # Add More Variables
    all_RVs['RV_sqrt'] = np.sqrt(all_RVs['RV'])
    all_RVs['log(RV)'] = np.log(all_RVs['RV'])

    all_RVs['C'] = all_RVs['RV'] - all_RVs[f'Jump_{alpha}']
    all_RVs.loc[all_RVs['C'] == 0, 'C'] = np.nan
    all_RVs['log(C)'] = np.log(all_RVs['C'])

    all_RVs['TC'] = all_RVs['RV'] - all_RVs[f'CTJump_{alpha}']
    all_RVs.loc[all_RVs['TC'] == 0, 'TC'] = np.nan
    all_RVs['log(TC)'] = np.log(all_RVs['TC'])

    # all_RVs['J_sqrt'] = np.sqrt(all_RVs[f'Jump_{alpha}'])
    all_RVs['log(J+1)'] = np.log(all_RVs[f'Jump_{alpha}'] + 1)
    # all_RVs.loc[all_RVs[f'Jump_{alpha}'] == 0, f'Jump_{alpha}'] = np.nan

    # all_RVs['TJ_sqrt'] = np.sqrt(all_RVs[f'CTJump_{alpha}'])
    all_RVs['log(TJ+1)'] = np.log(all_RVs[f'CTJump_{alpha}'] + 1)
    # all_RVs.loc[all_RVs[f'CTJump_{alpha}'] == 0, f'CTJump_{alpha}'] = np.nan

    drop_zero_col = [f'Jump_{alpha}', f'CTJump_{alpha}']

    summary_stats_1coin = SummaryStatistics(estimator_matrix=all_RVs,
                                            drop_zero_col=drop_zero_col,
                                            lower_percentile=0.05,
                                            higher_percentile=0.95,
                                            acf_lags=[1, 7, 30, 100])

    # Output statistics
    main_vars = ['RV', 'log(RV)', 'C', 'log(C)', 'TC', 'log(TC)', f'Jump_{alpha}',
                 'log(J+1)', f'CTJump_{alpha}', 'log(TJ+1)']
    # main_vars = ['RV', 'RV_sqrt', 'log(RV)', 'C', 'C_sqrt', 'log(C)', 'TC', 'TC_sqrt', 'log(TC)', f'Jump_{alpha}',
    #              'J_sqrt', 'log(J+1)', f'CTJump_{alpha}', 'TJ_sqrt', 'log(TJ+1)']
    auxi_vars = (summary_stats_1coin.columns) ^ (main_vars)
    main_var_statistics = summary_stats_1coin[main_vars]
    auxilinary_statistics = summary_stats_1coin[auxi_vars]

    for var in main_vars:
        main_var_statistics[var] = reformat_stats_number(main_var_statistics[var],
                                                         decimal_round=decimal,
                                                         format_threshold=format_threshold)

    for var in auxi_vars:
        auxilinary_statistics[var] = reformat_stats_number(auxilinary_statistics[var],
                                                           decimal_round=decimal,
                                                           format_threshold=format_threshold)

    # main_var_statistics.columns = ['RV1/2', 'log(RV)', 'C', 'TC', f'Jump(alpha)', f'TJump(alpha)']
    main_var_statistics.to_latex(stat_out_dir + f'Main_{coin}_{freq}_{alpha}_{cv}_{decimal}_{format_threshold}.csv')
    auxilinary_statistics.to_latex(stat_out_dir + f'Aux_{coin}_{freq}_{alpha}_{cv}_{decimal}_{format_threshold}.csv')

    # print(main_var_statistics.to_latex(f'test.csv'))
    # print(auxilinary_statistics.to_latex())

    return main_var_statistics, auxilinary_statistics


def indices_summarystat(decimal=2, format_threshold=0.0001):
    stat_out_dir = outdata_dir + f'SummaryStats/indices/'
    os.makedirs(stat_out_dir, exist_ok=True)

    # Read Data
    indices_rv = read_indices_full_RV()

    indices_rv_statsum = SummaryStatistics(indices_rv, acf_lags=[1, 7, 30, 100])

    for index in indices_rv.columns:
        indices_rv_statsum[index] = reformat_stats_number(indices_rv_statsum[index],
                                                          decimal_round=decimal,
                                                          format_threshold=format_threshold)

    indices_rv_statsum.to_csv(stat_out_dir + 'all_rv_SummaryStat.csv')
    indices_rv_statsum.to_latex(stat_out_dir + 'all_rv_summarystat_latex.csv')

    partial_rv = indices_rv_statsum[['.AEX', '.DJI', '.FTSE', '.HSI', '.SPX', '.SSEC']]
    partial_rv.to_latex(stat_out_dir + 'main_rv_summarystat_latex.csv')


def compute_summarystat():
    # coins = ['gemini_BTC', 'gemini_ETH', 'BTC', 'ETH', 'XRP', 'LTC']

    cv = cvs[0]
    freq = freqs[0]
    decimal = 2
    format_threshold = 0.0001

    for coin in coins:
        crypto_summarystats(coin=coin,
                            freq=freq,
                            cv=cv,
                            refresh=True,
                            alpha=alpha,
                            decimal=decimal,
                            format_threshold=format_threshold,
                            annualized=True,
                            truncate_zero=True)

compute_summarystat()
# indices_summarystat(decimal=2, format_threshold=0.0001)
