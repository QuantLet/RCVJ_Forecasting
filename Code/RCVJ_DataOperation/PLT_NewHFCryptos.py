import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt

root_path = os.getcwd()
data_path = root_path + '/Data/'

daily_close = pd.read_json(data_path + 'daily_close.json')
daily_close.set_index(keys='date', inplace=True, drop=True)



plt.figure(figsize=(15,5))
plt.plot(daily_close.loc[daily_close.index.date >= dt.date(2017,1,1), 'price'], color='blue')
plt.title('CRIX Index', fontsize=17)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('crix_daily_close.png', dpi=300)

btc_df = pd.read_csv(root_path + '/Data/BTC_USDT.csv', index_col=0, parse_dates=True)
eth_df = pd.read_csv(root_path + '/Data/ETH_USDT.csv', index_col=0, parse_dates=True)
xrp_df = pd.read_csv(root_path + '/Data/XRP_USDT.csv', index_col=0, parse_dates=True)
bch_df = pd.read_csv(root_path + '/Data/BCH_USDT.csv', index_col=0, parse_dates=True)
ltc_df = pd.read_csv(root_path + '/Data/LTC_USDT.csv', index_col=0, parse_dates=True)

def plot_hf(data):
    fig = plt.figure(figsize=(15,5))
    plt.plot(data, color='blue')
    plt.title('5-min frequency close price for Bitcoin')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    return fig

btc_fig = plot_hf(btc_df['close'])
btc_fig.savefig("bitcoin_hf.png", dpi=300)



btc_df['open'].plot()
btc_df['close'].plot()
