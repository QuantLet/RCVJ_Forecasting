from Code.GlobalParams import *
from Code.RCVJ_TimeSeriesAnalysis.HAR_TCJ_Models import HARTCJ_OLS_Estimation as har_ols
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
