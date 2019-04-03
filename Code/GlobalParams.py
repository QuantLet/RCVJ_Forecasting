import numpy as np
import os
import datetime as dt

np.random.seed(1124)
root_dir = os.getcwd()
data_dir = root_dir + '/Data/'
outdata_dir = root_dir + '/Output/'
outplot_dir = root_dir + '/Plot/'

# Global parameters
# coins = ['gemini_BTC', 'gemini_ETH', 'BTC', 'ETH', 'LTC', 'XRP', 'BCH']

coins = ['gemini_BTC', 'gemini_ETH']
freqs = ['5min']
cvs = [3]
alphas = [0.9999]
sample_num = 288
tz_lag = 0
split_date = dt.date(2018, 1, 1)
rv_names = ['RV', 'Log_RV', 'Sqrt_RV']  # 3 forms of RV
# filter_jumps = ['AllJump', 'ConJump', 'NoJump']  #
filter_jumps = ['AllJump']
horizons = [1, 7, 30]
expect_excess_returns = [0.05, 0.08]

model_names = ['HAR',
               'HAR_RVJ', 'HAR_RVJs',
               'HAR_RVTJ', 'HAR_RVTJs',
               'HAR_CJ', 'HAR_CJs',
               'HAR_TCJ', 'HAR_TCJs',
               'HAR_Exp_RV_J', 'HAR_Exp_RV_Js',
               'HAR_Exp_RV_TJ', 'HAR_Exp_RV_TJs',
               'HAR_Exp_CJ', 'HAR_Exp_CJs',
               'HAR_Exp_TCJ', 'HAR_Exp_TCJs']

# Delete some of the regression results, optional
col_to_del = [
    'HAR_RVJ',
    'HAR_RVTJ',
    'HAR_CJ',
    'HAR_TCJ',
    'HAR_Exp_RV_J',
    'HAR_Exp_RV_TJ',
    'HAR_Exp_CJ',
    'HAR_Exp_TCJ',
]
