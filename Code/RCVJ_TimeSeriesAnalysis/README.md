[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **RCVJ_TimeSeriesAnalysis** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: RCVJ_TimeSeriesAnalysis

Published in: Realized Cryptocurrencies Volatility Forecasting with Jumps

Description: This quantlet includes two time series analysis code files. The first one is the HAR forecasting model calibration, in-sample and out-of-sample forecasting. The second one is calculating the realized utility function documented on Bollerslev et al (2018).

Keywords: Realized volatility, Heterogenous Autoregression (HAR), Realized utility function, Cryptocurrencies, Global Market Indices, ACF

Author: Junjie Hu

Submitted: 03.04.2019

```

### PYTHON Code
```python

"""
This file is not for execution!
"""


"""
HAR_TCJ_Models.py
"""


import pandas as pd
import numpy as np
import datetime as dt
from Code.RCVJ_RealizedVolatility.CAL_RV_BPV import CAL_AllRVs
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from Code.GlobalParams import *
from itertools import product
import itertools
import pickle


def plot_forecast_errors(model, reg_df, rv_name, horizon, save_dir):
    # model = reg_har_rvjs
    forecast = model.predict(reg_df[model.model.exog_names[1:]])
    true = reg_df[f'{rv_name}_{horizon}']

    if rv_name == 'Log_RV':
        rv_hat = np.exp(forecast)
        true_rv = np.exp(true)
    elif rv_name == 'Sqrt_RV':
        rv_hat = np.power(forecast, 2)
        true_rv = np.power(true, 2)
    else:
        rv_hat = forecast
        true_rv = true

    errors = true_rv - rv_hat
    plt.plot(errors)

    fig = plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    plt.plot(true_rv.index, true_rv.values, color='blue', label='Ex-post RV')
    plt.plot(rv_hat.index, rv_hat.values, color='red', linestyle='--', label='Forecast RV')
    plt.xticks([])
    plt.ylabel('Realized Volatility', fontsize=17)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(errors, color='black', linestyle=(0, (1, 1)))
    plt.xlabel('Date', fontsize=17)
    plt.xticks(fontsize=14)
    plt.ylabel('Forecast Error', fontsize=17)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir, dpi=300)
    plt.close()
    # return fig


def HAR_forecast_evaluation(model, data_df, rv_name, horizon):
    """
    Function to evaluate the forecast resutls from HAR model
    Transform back to realized variance form to evaluate the performance
    :param model: Object of fitted regression model from statsmodels
    :param data_df: Dataframe type of all regressors and dependent variable
    :param rv_name: The form of realized variance
    :param horizon: Forecast horizon, used to recognize the model
    :return:
    """
    predict_result = model.predict(data_df[model.model.exog_names[1:]]).to_frame()
    true_rv = data_df[f'{rv_name}_{horizon}']
    compare_df = pd.concat([true_rv, predict_result], axis=1)
    compare_df.columns = ['RV_true', 'RV_pred']

    # Transform back Realized Variation
    if rv_name == 'Log_RV':
        compare_df = np.exp(compare_df)
    if rv_name == 'Sqrt_RV':
        compare_df = compare_df ** 2

    # Mincer-Zarnowitz regression
    mz_reg = f'RV_true ~ RV_pred'
    mz_result = smf.ols(mz_reg, data=compare_df).fit()
    mz_rsquare = mz_result.rsquared

    ## ========Plot the forecast and true
    # plt.figure(figsize=(15, 5))
    # plt.plot(compare_df['RV_true'])
    # plt.plot(compare_df['RV_pred'])
    # diff = compare_df['RV_true'] - compare_df['RV_pred']
    # plt.figure(figsize=(15, 5))
    # plt.plot(diff)
    ## ========finish plot

    # MSE
    mse = np.mean((compare_df['RV_true'] - compare_df['RV_pred']) ** 2)

    # HRMSE
    hrmse = np.sqrt(np.mean(((compare_df['RV_true'] - compare_df['RV_pred']) / compare_df['RV_true']) ** 2))

    # QLIKE
    qlike = np.mean(np.log(compare_df['RV_true']) + compare_df['RV_pred'] / compare_df['RV_true'])

    return predict_result, mz_rsquare, mse, hrmse, qlike


def construct_regressors(rv, c, jump, tc, t_jump, horizon, lags, rv_name, filter_jump, **kwargs):
    """
    Construct the regression data into the forecasting form
    """
    day_lag = lags[0]
    week_lag = lags[1]
    month_lag = lags[2]

    # shift backward 1 day, so to form the forecast
    # RV_f is the dependant variable on the left hand side of regression equation
    RV_f = rv.sort_index(axis=0, ascending=False).rolling(horizon).mean()
    RV_f.sort_index(axis=0, ascending=True, inplace=True)
    RV_f = RV_f.shift(-1)

    # All below are explanatory variables
    RV_day = rv.rolling(day_lag).mean()
    RV_week = rv.rolling(week_lag).mean()
    RV_month = rv.rolling(month_lag).mean()

    RV_Exp_day = rv.ewm(com=day_lag / 2).mean()
    RV_Exp_week = rv.ewm(com=week_lag / 2).mean()
    RV_Exp_month = rv.ewm(com=month_lag / 2).mean()

    C_day = c.rolling(day_lag).mean()
    C_week = c.rolling(week_lag).mean()
    C_month = c.rolling(month_lag).mean()

    C_exp_day = c.ewm(com=day_lag / 2).mean()
    C_exp_week = c.ewm(com=month_lag / 2).mean()
    C_exp_month = c.ewm(com=week_lag / 2).mean()

    Jump_day = jump.rolling(day_lag).mean()
    Jump_week = jump.rolling(week_lag).mean()
    Jump_month = jump.rolling(month_lag).mean()

    TC_day = tc.rolling(day_lag).mean()
    TC_week = tc.rolling(week_lag).mean()
    TC_month = tc.rolling(month_lag).mean()

    TC_exp_day = tc.ewm(com=day_lag / 2).mean()
    TC_exp_week = tc.ewm(com=month_lag / 2).mean()
    TC_exp_month = tc.ewm(com=week_lag / 2).mean()

    TJump_day = t_jump.rolling(day_lag).mean()
    TJump_week = t_jump.rolling(week_lag).mean()
    TJump_month = t_jump.rolling(month_lag).mean()

    reg_df = pd.concat([RV_f, RV_day, RV_week, RV_month,
                        RV_Exp_day, RV_Exp_week, RV_Exp_month,
                        C_day, C_week, C_month, C_exp_day, C_exp_week, C_exp_month,
                        Jump_day, Jump_week, Jump_month,
                        TC_day, TC_week, TC_month, TC_exp_day, TC_exp_week, TC_exp_month,
                        TJump_day, TJump_week, TJump_month], axis=1)

    # reg_df = pd.concat([RV_f, C_day, C_week, C_month, Jump_day, Jump_week, Jump_month], axis=1)
    reg_df.columns = [f'{rv_name}_{horizon}', f'{rv_name}_day', f'{rv_name}_week', f'{rv_name}_month',
                      f'{rv_name}_exp_day', f'{rv_name}_exp_week', f'{rv_name}_exp_month',
                      'C_day', 'C_week', 'C_month', 'C_exp_day', 'C_exp_week', 'C_exp_month',
                      'Jump_day', 'Jump_week', 'Jump_month',
                      'TC_day', 'TC_week', 'TC_month', 'TC_exp_day', 'TC_exp_week', 'TC_exp_month',
                      'TJump_day', 'TJump_week', 'TJump_month']
    reg_df.dropna(axis=0, inplace=True)
    reg_df = reg_df.round(10)
    reg_df = reg_df[~(reg_df['C_day'] == 0)]
    reg_df = reg_df[~(reg_df['TC_day'] == 0)]

    if kwargs['pred_split'] is not None:
        reg_df_train = reg_df[reg_df.index < kwargs['split_date']]
        reg_df_test = reg_df[reg_df.index >= kwargs['split_date']]
        if kwargs['pred_split'] == 'train':
            reg_df_out = reg_df_train
        elif kwargs['pred_split'] == 'test':
            reg_df_out = reg_df_test
        else:
            raise KeyError('split keyword error')
    else:
        reg_df_out = reg_df

    # Take different forms of RV and derived variables,
    # Take logarithm and square root of variables
    if rv_name == 'Log_RV':
        log_1_vars = [f'{rv_name}_{horizon}', f'{rv_name}_day', f'{rv_name}_week', f'{rv_name}_month',
                      f'{rv_name}_exp_day', f'{rv_name}_exp_week', f'{rv_name}_exp_month',
                      'C_day', 'C_week', 'C_month', 'C_exp_day', 'TC_exp_week', 'TC_exp_month',
                      'TC_day', 'TC_week', 'TC_month', 'TC_exp_day', 'TC_exp_week', 'TC_exp_month', ]
        log_2_vars = ['Jump_day', 'Jump_week', 'Jump_month', 'TJump_day', 'TJump_week', 'TJump_month']
        print('Logarithm values')
        reg_df_out[log_1_vars] = np.log(reg_df_out[log_1_vars])
        reg_df_out[log_2_vars] = np.log(reg_df_out[log_2_vars] + 1)

    if rv_name == 'Sqrt_RV':
        print('Square root of values')
        reg_df_out = np.sqrt(reg_df_out)

    # filter_jumps = ['AllJump', 'ConJump', 'NoJump']
    # return reg_df_out_all, reg_df_out_conjump, reg_df_out_nojump
    if filter_jump == 'AllJump':
        return reg_df_out
    elif filter_jump == 'ConJump':
        return reg_df_out[reg_df_out[f'Jump_day'] != 0]
    elif filter_jump == 'NoJump':
        return reg_df_out[reg_df_out[f'Jump_day'] == 0]
    else:
        print('Keyword Not Available')
        raise KeyError


def write_results_table(result_df, model, model_name, pred_result):
    coef = model.params
    t_values = model.tvalues
    t_index = [ind + '_tvalue' for ind in coef.index]
    result_df.loc[coef.index, model_name] = coef.values
    result_df.loc[t_index, model_name] = t_values.values
    pred_result.insert(0, model.rsquared)
    result_df.loc[['Adj-R2', 'M-Z-R2', 'MSE', 'HRMSE', 'QLIKE'], model_name] = pred_result
    return result_df


def HAR_fit(reg_df, horizon, rv_name):
    """
    h = forecast_horizon
    h=1 means forecast on the next day
    forecast the rv on day t using info before day t (start from day t-1), full regression formula as follow:
    rv_t:t+h-1 = a0 +
                b_d * c_(t-lags[0]):t-1 +
                b_w * c_(t-lags[1]):t-1 +
                b_m * c_(t-lags[2]):t-1 +
                b_jd * j_(t-lags[0]):t-1 +
                # b_jw * j_(t-lags[1]):t-1 +
                # b_jm * j_(t-lags[2]):t-1 +
                epsilon_t

    :param reg_df:
    :param horizon:
    :param lags:
    :param threshold_jump:
    :param rv_name:
    :param out_dir:
    :param filter_jumps:
    :return:
    """

    """
    Construct the regression data into the forecasting form
    """

    # === Choose mac lag for HAC
    # [7, 14, 60]
    if horizon == 1:
        hac_max_lag = 7
    elif horizon == 7:
        hac_max_lag = 14
    elif horizon == 30:
        hac_max_lag = 60
    else:
        # print('Forecast horizon needs to be fixed')
        raise ValueError('Forecast horizon needs to be fixed')

    """
    Regression formula
    """
    HAR = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month'

    HAR_RVJ = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month + Jump_day'
    HAR_RVJs = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month + Jump_day + Jump_week + Jump_month'

    HAR_RVTJ = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month + TJump_day'
    HAR_RVTJs = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month + TJump_day + TJump_week + TJump_month'

    HAR_CJ = f'{rv_name}_{horizon} ~ C_day + C_week+ C_month + Jump_day'
    HAR_CJs = f'{rv_name}_{horizon} ~ C_day + C_week+ C_month + Jump_day + Jump_day + Jump_week + Jump_month'

    HAR_TCJ = f'{rv_name}_{horizon} ~ TC_day + TC_week+ TC_month + TJump_day'
    HAR_TCJs = f'{rv_name}_{horizon} ~ TC_day + TC_week+ TC_month + TJump_day + TJump_week + TJump_month'

    HAR_Exp_RV_J = f'{rv_name}_{horizon} ~ {rv_name}_exp_day + {rv_name}_exp_week+ {rv_name}_exp_month + Jump_day'
    HAR_Exp_RV_Js = f'{rv_name}_{horizon} ~ {rv_name}_exp_day + {rv_name}_exp_week+ {rv_name}_exp_month + Jump_day + Jump_week + Jump_month'

    HAR_Exp_RV_TJ = f'{rv_name}_{horizon} ~ {rv_name}_exp_day + {rv_name}_exp_week+ {rv_name}_exp_month + TJump_day'
    HAR_Exp_RV_TJs = f'{rv_name}_{horizon} ~ {rv_name}_exp_day + {rv_name}_exp_week+ {rv_name}_exp_month + TJump_day + TJump_week + TJump_month'

    HAR_Exp_CJ = f'{rv_name}_{horizon} ~ C_exp_day + C_exp_week+ C_exp_month + Jump_day'
    HAR_Exp_CJs = f'{rv_name}_{horizon} ~ C_exp_day + C_exp_week+ C_exp_month + Jump_day + Jump_week + Jump_month'

    HAR_Exp_TCJ = f'{rv_name}_{horizon} ~ TC_exp_day + TC_exp_week+ TC_exp_month + TJump_day'
    HAR_Exp_TCJs = f'{rv_name}_{horizon} ~ TC_exp_day + TC_exp_week+ TC_exp_month + TJump_day + TJump_week + TJump_month'

    # =======test for data quality
    # for col in reg_df.columns:
    # nan_values = reg_df[reg_df[col].isna()]
    # plt.figure(figsize=(15,5))
    # plt.plot(reg_df[col])
    # plt.title(col)
    # print(col,nan_values)

    # OLS estimation using HAC (Newey-West correction on residual term), max lags = month lags + dependent variable lags
    # a = smf.ols(HAR_TCJ, data=reg_df).fit(cov_type='HAC', cov_kwds={'maxlags': hac_max_lag})
    # a.summary()
    # ===========Finish test

    """
    Fit regression model
    """
    reg_har, \
    reg_har_rvj, reg_har_rvjs, \
    reg_har_rvtj, reg_har_rvtjs, \
    reg_har_cj, reg_har_cjs, \
    reg_har_tcj, reg_har_tcjs, \
    reg_har_exp_rv_j, reg_har_exp_rv_js, \
    reg_har_exp_rv_tj, reg_har_exp_rv_tjs, \
    reg_har_exp_cj, reg_har_exp_cjs, \
    reg_har_exp_tcj, reg_har_exp_tcjs = list(
        map(lambda formula: smf.ols(formula, data=reg_df).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_max_lag}),
            (HAR,
             HAR_RVJ, HAR_RVJs,
             HAR_RVTJ, HAR_RVTJs,
             HAR_CJ, HAR_CJs,
             HAR_TCJ, HAR_TCJs,
             HAR_Exp_RV_J, HAR_Exp_RV_Js,
             HAR_Exp_RV_TJ, HAR_Exp_RV_TJs,
             HAR_Exp_CJ, HAR_Exp_CJs,
             HAR_Exp_TCJ, HAR_Exp_TCJs)))

    """
    Regression results
    """
    # print(reg_har.summary())
    reg_har.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M'])

    # print(reg_har_rvj.summary())
    reg_har_rvj.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD'])

    # print(reg_har_rvjs.summary())
    reg_har_rvjs.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM'])

    # print(reg_har_rvtj.summary())
    reg_har_rvtj.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD'])

    # print(reg_har_rvtjs.summary())
    reg_har_rvtjs.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM'])

    # print(reg_har_cj.summary())
    reg_har_cj.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD'])

    # print(reg_har_cjs.summary())
    reg_har_cjs.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM'])

    # print(reg_har_tcj.summary())
    reg_har_tcj.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD'])

    # print(reg_har_tcjs.summary())
    reg_har_tcjs.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM'])

    # print(reg_har_exp_rv_j.summary())
    reg_har_exp_rv_j.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD'])

    # print(reg_har_exp_rv_js.summary())
    reg_har_exp_rv_js.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM'])

    # print(reg_har_exp_rv_tj.summary())
    reg_har_exp_rv_tj.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD'])

    # print(reg_har_exp_rv_tjs.summary())
    reg_har_exp_rv_tjs.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM'])

    # print(reg_har_exp_cj.summary())
    reg_har_exp_cj.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD'])

    # print(reg_har_exp_cjs.summary())
    reg_har_exp_cjs.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM'])

    # print(reg_har_exp_tcj.summary())
    reg_har_exp_tcj.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD'])

    # print(reg_har_exp_tcjs.summary())
    reg_har_exp_tcjs.summary(xname=['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM'])

    return reg_har, reg_har_rvj, reg_har_rvjs, reg_har_rvtj, reg_har_rvtjs, reg_har_cj, reg_har_cjs, \
           reg_har_tcj, reg_har_tcjs, reg_har_exp_rv_j, reg_har_exp_rv_js, reg_har_exp_rv_tj, \
           reg_har_exp_rv_tjs, reg_har_exp_cj, reg_har_exp_cjs, reg_har_exp_tcj, reg_har_exp_tcjs


class HARTCJ_OLS_Estimation:

    def __init__(self, coin, freq, alpha, cv, tz_lag, rv_name, filter_jump, horizon, split_date):
        self.coin = coin
        self.freq = freq
        self.alpha = alpha
        self.cv = cv
        self.tz_lag = tz_lag
        self.rv_name = rv_name
        self.filter_jump = filter_jump
        self.horizon = horizon
        # self.pred_split = pred_split
        self.split_date = split_date

        # fixed arguments
        self.refresh, self.refresh_est, self.truncate_zero, self.annualized = True, False, True, True
        self.type_name = '_'.join([coin, str(cv), str(alpha), str(tz_lag)])
        self.version_name = '_'.join([rv_name, str(horizon), freq, filter_jump, split_date.strftime('%Y%m%d')])
        print(f'TYPE: {self.type_name}')
        print(f'VERSION: {self.version_name}')
        # Lags for daily, weekly and monthly
        self.lags = [1, 7, 30]

    def read_all_variables(self):
        """
        Read all variables from RV estimate function
        :return: RV, J, TJ, C, TC, LogReturn
        """
        sample_num = 288
        all_RVs, estimation, self.logreturn = CAL_AllRVs(coin=self.coin,
                                                         freq=self.freq,
                                                         sample_num=sample_num,
                                                         refresh=self.refresh,
                                                         refresh_est=self.refresh_est,
                                                         cv=self.cv,
                                                         truncate_zero=self.truncate_zero,
                                                         alpha=self.alpha,
                                                         annualized=self.annualized,
                                                         tz_lag=self.tz_lag)

        self.rv = all_RVs['RV']
        self.jump = all_RVs[f'Jump_{self.alpha}']
        self.jump[self.jump < 0] = 0
        self.t_jump = all_RVs[f'CTJump_{self.alpha}']
        self.t_jump[self.t_jump < 0] = 0
        self.c = self.rv - self.jump
        self.tc = self.rv - self.t_jump
        return self.rv, self.jump, self.t_jump, self.c, self.tc, self.logreturn

    def forecast_evaluate_record(self, pred_split, plot_error):

        """

        :param self.models:
        :param reg_df:
        :param rv_name:
        :param filter_jump:
        :param horizon:
        :return:
        """

        """
        HAR = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month'

        HAR_RVJ = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month + Jump_day'
        HAR_RVJs = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month + Jump_day + Jump_week + Jump_month'

        HAR_RVTJ = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month + TJump_day'
        HAR_RVTJs = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month + TJump_day + TJump_week + TJump_month'

        HAR_CJ = f'{rv_name}_{horizon} ~ C_day + C_week+ C_month + Jump_day'
        HAR_CJs = f'{rv_name}_{horizon} ~ C_day + C_week+ C_month + Jump_day + Jump_day + Jump_week + Jump_month'

        HAR_TCJ = f'{rv_name}_{horizon} ~ TC_day + TC_week+ TC_month + TJump_day'
        HAR_TCJs = f'{rv_name}_{horizon} ~ TC_day + TC_week+ TC_month + TJump_day + TJump_week + TJump_month'

        HAR_RV_Exp_CJ = f'{rv_name}_{horizon} ~ {rv_name}_exp_day + {rv_name}_exp_week+ {rv_name}_exp_month + Jump_day'
        HAR_RV_Exp_CJs = f'{rv_name}_{horizon} ~ {rv_name}_exp_day + {rv_name}_exp_week+ {rv_name}_exp_month + Jump_day + Jump_week + Jump_month'

        HAR_RV_Exp_TCJ = f'{rv_name}_{horizon} ~ {rv_name}_exp_day + {rv_name}_exp_week+ {rv_name}_exp_month + TJump_day'
        HAR_RV_Exp_TCJs = f'{rv_name}_{horizon} ~ {rv_name}_exp_day + {rv_name}_exp_week+ {rv_name}_exp_month + TJump_day + TJump_week + TJump_month'

        HAR_Exp_CJ = f'{rv_name}_{horizon} ~ C_exp_day + C_exp_week+ C_exp_month + Jump_day'
        HAR_Exp_CJs = f'{rv_name}_{horizon} ~ C_exp_day + C_exp_week+ C_exp_month + Jump_day + Jump_week + Jump_month'

        HAR_Exp_TCJ = f'{rv_name}_{horizon} ~ TC_exp_day + TC_exp_week+ TC_exp_month + TJump_day'
        HAR_Exp_TCJs = f'{rv_name}_{horizon} ~ TC_exp_day + TC_exp_week+ TC_exp_month + TJump_day + TJump_week + TJump_month'
        """

        """
        model_names = ['HAR',
                       'HAR_RVJ', 'HAR_RVJs',
                       'HAR_RVTJ', 'HAR_RVTJs',
                       'HAR_CJ', 'HAR_CJs',
                       'HAR_TCJ', 'HAR_TCJs',
                       'HAR_Exp_RV_J', 'HAR_Exp_RV_Js',
                       'HAR_Exp_RV_TJ', 'HAR_Exp_RV_TJs',
                       'HAR_Exp_CJ', 'HAR_Exp_CJs',
                       'HAR_Exp_TCJ', 'HAR_Exp_TCJs']

        """
        if pred_split == 'train':
            reg_df = self.in_sample_reg_df
        elif pred_split == 'test':
            reg_df = self.out_sample_reg_df
        else:
            raise KeyError('Split keyword error')

        index_coef = ['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM']
        index_t = [ind + '_tvalue' for ind in index_coef]
        indexs = index_coef + index_t + ['Adj-R2', 'M-Z-R2', 'MSE', 'HRMSE', 'QLIKE']
        self.reg_metric_results = pd.DataFrame(columns=model_names, index=indexs)
        # models = self.models
        model_params = list(zip(self.models, model_names))

        self.rv_pred = pd.DataFrame()  # rv_pred is the forecast realized volatility

        for model, model_name in model_params:
            # print(model)
            print(model_name)

            predict_result, *har_pred_result = HAR_forecast_evaluation(model=model, data_df=reg_df,
                                                                       rv_name=self.rv_name,
                                                                       horizon=self.horizon)
            predict_result.columns = [model_name]
            self.rv_pred = pd.concat([self.rv_pred, predict_result], axis=1)

            if plot_error is True:
                error_dir = outplot_dir + f'ForecastError/{self.type_name}/'
                os.makedirs(error_dir, exist_ok=True)
                error_path = f'{error_dir}/{self.version_name}_{pred_split}.png'
                plot_forecast_errors(model=model, reg_df=reg_df, rv_name=self.rv_name, horizon=self.horizon,
                                     save_dir=error_path)

            independ_var_num = len(model.params)
            if independ_var_num == 4:
                model.model.data.xnames = ['alpha', 'beta_D', 'beta_W', 'beta_M']
            elif independ_var_num == 5:
                model.model.data.xnames = ['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD']
            elif independ_var_num == 7:
                model.model.data.xnames = ['alpha', 'beta_D', 'beta_W', 'beta_M', 'beta_JD', 'beta_JW', 'beta_JM']

            write_results_table(self.reg_metric_results, model, model_name, list(har_pred_result))

        self.reg_metric_results.sort_index(axis=0, ascending=True, inplace=True)
        self.reg_metric_results = self.reg_metric_results.astype(float).round(3)


        stats_output_dir = outdata_dir + f'HAR_Summary/{self.type_name}/'
        print(stats_output_dir)
        os.makedirs(stats_output_dir, exist_ok=True)

        self.reg_metric_results.drop(labels=col_to_del, axis=1, inplace=True)
        self.reg_metric_results.to_csv(stats_output_dir + f'{self.version_name}_{pred_split}.csv')
        self.reg_metric_results.to_latex(stats_output_dir + f'{self.version_name}_{pred_split}.csv')

        if self.rv_name == 'Log_RV':
            reg_df.loc[:, f'{self.rv_name}_{self.horizon}'] = np.exp(reg_df.loc[:, f'{self.rv_name}_{self.horizon}'])
            self.rv_pred = np.exp(self.rv_pred)
        elif self.rv_name == 'Sqrt_RV':
            reg_df.loc[:, f'{self.rv_name}_{self.horizon}'] = np.power(reg_df.loc[:, f'{self.rv_name}_{self.horizon}'],
                                                                       2)
            self.rv_pred = np.power(self.rv_pred, 2)

        self.rv_pred['rv_true'] = reg_df[f'{self.rv_name}_{self.horizon}']
        self.rv_pred.to_csv(stats_output_dir + f'PredResult_{self.version_name}_{pred_split}.csv')
        return self.rv_pred

    def model_fit(self, save):

        """
        Versions include if condition on jumps, the form of RV
        :param rv_name:
        :param filter_jump:
        :param horizon:
        :return:
        """

        # construct all variables
        """
        return data set:
        full regression entries
        only entries with significant jumps
        only entries with no jumps
        only entries with significant threshold-jumps
        only entries with no threshold-jumps
        """

        # Check if RV variables are read
        if not all(hasattr(self, var) for var in ['rv', 'c', 'jump', 'tc', 't_jump', 'logreturn']):
            self.read_all_variables()

        # self.rv, self.jump, self.t_jump, self.c, self.tc, self.logreturn

        # pred_split = 'train'
        self.in_sample_reg_df = construct_regressors(
            rv=self.rv,
            c=self.c,
            jump=self.jump,
            tc=self.tc,
            t_jump=self.t_jump,
            horizon=self.horizon,
            lags=self.lags,
            rv_name=self.rv_name,
            filter_jump=self.filter_jump,
            pred_split='train',
            split_date=self.split_date)

        print(f'In-sample Fit Period: {self.in_sample_reg_df.index[0]} - {self.in_sample_reg_df.index[-1]}')

        # In sample fit models
        self.models = HAR_fit(reg_df=self.in_sample_reg_df, horizon=self.horizon, rv_name=self.rv_name)

        if save:
            models_dir = outdata_dir + f'HAR_models/{self.type_name}/'
            os.makedirs(models_dir, exist_ok=True)

            # evaluate forecasting power and write results to csv file

            with open(models_dir + f'{self.version_name}_train.pkl', 'wb') as model_file:
                pickle.dump(self.models, model_file)

        return self.models

    def insample_forecast(self, save, plot_error):
        self.model_fit(save=save)
        # Evaluate the in sample forecast results
        self.rv_pred = self.forecast_evaluate_record(pred_split='train', plot_error=plot_error)

    def out_sample_forecast(self, save, plot_error):
        """

        :return:
        """
        self.model_fit(save=save)
        # pred_split = 'test'
        self.out_sample_reg_df = construct_regressors(
            rv=self.rv,
            c=self.c,
            jump=self.jump,
            tc=self.tc,
            t_jump=self.t_jump,
            horizon=self.horizon,
            lags=self.lags,
            rv_name=self.rv_name,
            filter_jump=self.filter_jump,
            pred_split='test',
            split_date=self.split_date)

        print(f'Out-of Sample Forecast Period: {self.out_sample_reg_df.index[0]} - {self.out_sample_reg_df.index[-1]}')

        # Evaluate the our of sample forecast results
        self.rv_pred = self.forecast_evaluate_record(pred_split='test', plot_error=plot_error)

        return self.rv_pred


def main_HAR_estimate_func():
    # === ADJUST GLOBAL PARAMETERS IN FILE: GlobalParams.py

    params = itertools.product(coins, freqs, alphas, cvs, rv_names, filter_jumps, horizons)

    for coin, freq, alpha, cv, rv_name, filter_jump, horizon in params:
        estimator = HARTCJ_OLS_Estimation(coin=coin, freq=freq, alpha=alpha, cv=cv, tz_lag=tz_lag, rv_name=rv_name,
                                          filter_jump=filter_jump, horizon=horizon, split_date=split_date)
        estimator.insample_forecast(save=True, plot_error=True)
        estimator.out_sample_forecast(save=True, plot_error=True)

    # self.record_results(self.models, self.reg_df, rv_name, filter_jump, horizon)


def quick_test_func():
    # Quick Test
    # coins = ['gemini_BTC', 'gemini_ETH', 'BTC', 'ETH']
    # # , 'XRP', 'LTC'
    # freqs = ['5min']
    # alphas = [0.9999]
    # cvs = [1, 2, 3, 4, 5]

    coin = coins[0]
    freq = freqs[0]
    # alpha = alphas[0]
    cv = cvs[0]
    horizons = [1, 7, 30]
    rv_names = ['RV', 'Log_RV', 'Sqrt_RV']
    filter_jumps = ['AllJump']
    filter_jump = filter_jumps[0]
    horizon = horizons[0]
    rv_name = rv_names[1]
    lags = [1, 7, 30]
    tz_lag = 0
    # pred_split = 'train'
    split_date = dt.date(2018, 1, 1)

    HAR_models = HARTCJ_OLS_Estimation(coin=coin, freq=freq, alpha=alpha, cv=cv, tz_lag=tz_lag, rv_name=rv_name,
                                       filter_jump=filter_jump, horizon=horizon, split_date=split_date)
    # HAR_models.refresh = False

    rv, jump, t_jump, c, tc, logreturn = HAR_models.read_all_variables()

    # test regressors construction

    in_sample_reg = construct_regressors(rv=rv, c=c, tc=tc, jump=jump, t_jump=t_jump, horizon=horizon, lags=lags,
                                         rv_name=rv_name, filter_jump=filter_jump, pred_split='train',
                                         split_date=split_date)

    HAR = f'{rv_name}_{horizon} ~ {rv_name}_day + {rv_name}_week + {rv_name}_month'
    HAR_C = f'{rv_name}_{horizon} ~ C_day + C_week+ C_month'
    HAR_CJs = f'{rv_name}_{horizon} ~ C_day + C_week+ C_month + Jump_day + Jump_day + Jump_week + Jump_month'
    hac_max_lag = 7

    #
    # out_sample_reg = construct_regressors(rv=rv, c=c, tc=tc, jump=jump, t_jump=t_jump, horizon=horizon, lags=lags,
    #                                       rv_name=rv_name, filter_jump=filter_jump, pred_split='test',
    #                                       split_date=split_date)

    # har_models = HAR_models.model_fit(save=True)

    # HAR_models.model_fit(save=True)

    HAR_models.insample_forecast(save=True, plot_error=True)

    rv_pred = HAR_models.out_sample_forecast(save=True, plot_error=True)

    reg_har = smf.ols(HAR, data=in_sample_reg).fit(cov_type='HAC', cov_kwds={'maxlags': hac_max_lag})
    reg_harc = smf.ols(HAR_C, data=in_sample_reg).fit(cov_type='HAC', cov_kwds={'maxlags': hac_max_lag})
    reg_harcjs = smf.ols(HAR_CJs, data=in_sample_reg).fit(cov_type='HAC', cov_kwds={'maxlags': hac_max_lag})

    print(reg_har.summary())
    print(reg_harc.summary())
    print(reg_harcjs.summary())


# Execute
if __name__ == '__main__':
    main_HAR_estimate_func()




"""
CAL_UtilityFunction.py
"""

from Code.GlobalParams import * from Code.RCVJ_TimeSeriesAnalysis.HAR_TCJ_Models import HARTCJ_OLS_Estimation as har_ols
import datetime as dt
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
import itertools


class CAL_UtilityFunction:

    def __init__(self, coin, freq, alpha, cv, rv_name, filter_jump, horizon, tz_lag, expect_excess_return, pred_split,
                 split_date):
        """

        :param coin:
        :param freq:
        :param alpha:
        :param cv:
        """

        #  Control of different of versions
        self.coin = coin
        self.freq = freq
        self.alpha = alpha
        self.cv = cv
        self.rv_name = rv_name
        self.filter_jump = filter_jump
        self.horizon = horizon
        self.tz_lag = tz_lag
        self.refresh = True
        self.annualized = True
        self.pred_split = pred_split
        self.split_date = split_date
        self.type_name = '_'.join([coin, str(cv), str(alpha), str(tz_lag), str(expect_excess_return)])
        self.version_name = '_'.join(
            [rv_name, str(horizon), freq, filter_jump, pred_split, split_date.strftime('%Y%m%d')])
        print(f'TYPE: {self.type_name}')
        print(f'VERSION: {self.version_name}')

        # self.rolling_win_size = rolling_win_size

        # Output dir
        self.utility_output_dir = outdata_dir + f'UtilityEvaluationModels/{self.type_name}/'
        print(self.utility_output_dir)
        os.makedirs(self.utility_output_dir, exist_ok=True)

        # Parameters
        # Expect excess return = sr^2 / gamma, sr=1, gamma=2
        self.expect_excess_return = expect_excess_return

        # self.gamma = 10
        # self.r_f = 0.06
        # self.models_ind = {name:num for name, num in zip(model_names,range(0,17))}

    # Functions to estimate models

    def preprocess_data(self, read_results=True):
        """

        :param rv_name:
        :param filter_jump:
        :param horizon:
        :return:
        """
        self.har_estimation = har_ols(coin=self.coin, freq=self.freq, alpha=self.alpha, cv=self.cv, tz_lag=self.tz_lag,
                                      rv_name=self.rv_name, filter_jump=self.filter_jump, horizon=self.horizon,
                                      split_date=self.split_date)  # HAR instance

        # self.har_estimation.annualized = self.annualized
        self.models = self.har_estimation.model_fit(save=False)  # Fit the model

        if self.pred_split == 'train':
            self.rv_pred = self.har_estimation.insample_forecast(save=False, plot_error=False)
            self.rv_true = self.rv_pred['rv_true'].to_frame()
        elif self.pred_split == 'test':
            self.rv_pred = self.har_estimation.out_sample_forecast(save=False, plot_error=False)
            self.rv_true = self.rv_pred['rv_true'].to_frame()

    def calculate_realized_utility(self):
        self.uow_t = pd.DataFrame(columns=model_names)
        # self.uow = pd.DataFrame(columns=model_names)
        for model_name in self.uow_t.columns:
            ru = realized_utility(exp_exreturn=self.expect_excess_return,
                                  RV_exp=self.rv_pred[model_name],
                                  rv_t1=self.rv_true)
            self.uow_t.loc[:, model_name] = ru
        true_ru = realized_utility(exp_exreturn=self.expect_excess_return,
                                   RV_exp=self.rv_pred['rv_true'],
                                   rv_t1=self.rv_true)
        self.uow_t['ru_true'] = true_ru
        self.uow = self.uow_t.mean()

    def save_results(self):
        uow_file_path = f'{self.utility_output_dir}/{self.version_name}'
        with open(uow_file_path + '.pkl', 'wb') as uow_t_file:
            pickle.dump(self.uow_t, uow_t_file)
        # self.uow = self.uow.to_frame()
        self.uow.to_csv(uow_file_path + '.csv')


def realized_utility(exp_exreturn, RV_exp, rv_t1):
    """
    Calculate realized utility (Bollerslev (2018))
    uow = mean(SR^2/gamma * sqrt(RV_t+1 / RV_forecast) - SR^2/2gamma * RV_t+1 / RV_forecast)
    :param exp_exreturn: expected excessive return, SR^2/gamma
    :param RV_exp: expected T+1 day realized volatility at day T
    :param rv_t1: Realized volatility at day T+1
    :return:
    """
    # RV_exp = RV_exp.shift(1)
    uow_df = pd.concat([RV_exp, rv_t1], axis=1, sort=True)
    uow_df.columns = ['rv_pred', 'rv_t1']
    uow_df.dropna(inplace=True)
    uow_t = exp_exreturn * np.divide(np.sqrt(uow_df['rv_t1']), np.sqrt(uow_df['rv_pred'])) - (
            exp_exreturn / 2) * np.divide(uow_df['rv_t1'], uow_df['rv_pred'])
    # uow =uow_t.mean()
    return uow_t


def optimal_weight(r_exp, r_f, gamma, RV_exp, col_name):
    """
    :param r_exp: rolling window returns
    :param r_f: risk-free rate
    :param gamma: relative risk aversion coefficient
    :param RV_exp: Forecast values to RV
    :return: optimal weight for next day (shift(1)) for different models, DataFrame
    """
    """
    Optimal weight from maximum utility function
    w = (E(r_t+1) - r_f_t+1) / (gamma * RV(r_t+1))
    """
    w_c = pd.DataFrame(columns=col_name)
    for col in RV_exp.columns:
        # col = 'HAR'
        w_c.loc[:, col] = (r_exp - r_f) / (gamma * RV_exp[col])

    w_c[w_c > 1] = 1
    w_c[w_c < 0] = 0

    # shift to next day to calculate utility
    w_c = w_c.shift(1)
    w_c.dropna(inplace=True)

    return w_c


def annualized_dailyreturn(logreturn_hf, window_size):
    """
    transform high freq log-returns to rolling window annualized daily return
    :return:
    """
    daily_logreturn = logreturn_hf.groupby(by=logreturn_hf.index.date, axis=0, sort=True).apply(lambda x: x.sum())
    daily_return = np.exp(daily_logreturn) - 1
    # annual_daily_return = (daily_return +1) ** 365 - 1
    rolling_daily_return = daily_return.rolling(window=window_size).mean()
    # rolling_annual_return = (rolling_daily_return + 1) ** 365 - 1
    return daily_return, rolling_daily_return


def average_utility(weight_f_t1, rv_t, r_t, gamma, r_f):
    """
    Given weight series forecast on day t+1, compute the average utility
    :param weight_f_t1: weight series for risky asset on day t+1
    :param rv_t: realized volatility on day t
    :param r_t: daily return on day t
    :param gamma: relative risk aversion coefficient
    :param r_f: risk-free rate
    :return:
    """
    """
    Average_Utility(p) = mean(premium return_t+1 - gamma/2 * rv+1)
    premium return_t+1 = weight_series * r_t+1 + (1-weight_series) * r_f
    """

    utility_df = pd.concat([weight_f_t1, rv_t, r_t], axis=1, sort=True)
    utility_df.columns = ['weight', 'rv', 'r']
    utility_df.dropna(inplace=True)

    prem_r = utility_df.weight * utility_df.r + (1 - utility_df.weight) * r_f
    average_utility = prem_r - (gamma / 2) * (np.multiply(utility_df.rv, utility_df.weight ** 2))
    # average_utility = prem_r - (gamma/2) * utility_df.rv
    # average_utility = prem_r - (gamma/(2*(1+gamma))) * utility_df.rv
    # average_utility = prem_r - (gamma/2) * (prem_r**2)
    average_utility.dropna(inplace=True)
    average_utility = average_utility.mean()
    return average_utility


def main_func():
    params = itertools.product(coins, freqs, cvs, rv_names, alphas, filter_jumps, horizons, expect_excess_returns)
    for coin, freq, cv, rv_name, alpha, filter_jump, horizon, expect_excess_return in params:
        utility_estimate = CAL_UtilityFunction(coin=coin,
                                               freq=freq,
                                               alpha=alpha,
                                               cv=cv,
                                               rv_name=rv_name,
                                               filter_jump=filter_jump,
                                               horizon=horizon,
                                               tz_lag=0,
                                               expect_excess_return=expect_excess_return,
                                               pred_split='test',  # adjust as testing mode
                                               split_date=split_date)
        utility_estimate.preprocess_data()
        utility_estimate.calculate_realized_utility()  # Calculate realized utility
        utility_estimate.save_results()  # Save realized utility results
        uow_t = utility_estimate.uow_t
        uow = utility_estimate.uow
        print(uow_t)
        print(uow)


def test_func():
    # Quick Test
    coin_G = coins[0]
    cv = cvs[0]
    rv_name = 'Log_RV'
    filter_jump = 'AllJump'
    freq = freqs[0]
    horizon = horizons[0]
    alpha = alphas[0]

    utility_estimate = CAL_UtilityFunction(coin=coin_G,
                                           freq=freq,
                                           alpha=alpha,
                                           cv=cv,
                                           rv_name=rv_name,
                                           filter_jump=filter_jump,
                                           horizon=horizon,
                                           tz_lag=0,
                                           expect_excess_return=0.08,
                                           pred_split='train',
                                           split_date=dt.date(2018, 1, 1))

    utility_estimate.preprocess_data()

    # utility_estimate.reg_df
    # utility_estimate.forecast_rv()
    utility_estimate.calculate_realized_utility()
    utility_estimate.save_results()
    uow_t = utility_estimate.uow_t
    uow = utility_estimate.uow
    print(uow_t)
    print(uow)


if __name__ == '__main__':
    main_func()


```

automatically created on 2019-04-03