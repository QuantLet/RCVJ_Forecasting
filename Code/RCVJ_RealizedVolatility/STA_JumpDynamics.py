import pandas as pd
from Code.GlobalParams import *
from Code.RCVJ_RealizedVolatility.CAL_RV_BPV import CAL_AllRVs



def jump_summary_statistics(coin, freq, cv=3, refresh=True, backup_data=True, alpha=0.99, annualized=True):
    # TODO: Jumps dynamics with different alphas and cvs for different coins

    # coin = 'gemini_BTC'
    # freq = '5min'
    # cv = 3
    # refresh = True
    # backup_data = True
    # alpha = 0.99

    alphas = [0.5, 0.95, 0.99, 0.999, 0.9999]
    cvs = [1, 2, 2.5, 3, 3.5, 4, 5]

    stats_out_dir = outdata_dir + 'StatisticsResults/'
    os.makedirs(stats_out_dir, exist_ok=True)

    all_RVs, estimation, logreturn = CAL_AllRVs(coin=coin,
                                                freq=freq,
                                                refresh=refresh,
                                                refresh_est=False,
                                                backup_data=backup_data,
                                                cv=cv,
                                                truncate_zero=False,
                                                alpha=alpha,
                                                annualized=annualized)

