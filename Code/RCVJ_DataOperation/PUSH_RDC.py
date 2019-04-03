import pandas as pd
# import mysql.connector
import peewee
import os

root_path = os.getcwd()
data_path = root_path + '/Data/'

conn_params = {'host': 'crix.hu-berlin.de',
               'port': 3306,
               'user': 'junjie',
               'password': f"\\ \'Kn^6kT"}


foo = r'baz "\"'
print(foo)


btc_df = pd.read_csv(root_path + '/Data/BTC_USDT.csv', index_col=0, parse_dates=True)
eth_df = pd.read_csv(root_path + '/Data/ETH_USDT.csv', index_col=0, parse_dates=True)
xrp_df = pd.read_csv(root_path + '/Data/XRP_USDT.csv', index_col=0, parse_dates=True)
ltc_df = pd.read_csv(root_path + '/Data/LTC_USDT.csv', index_col=0, parse_dates=True)
bch_df = pd.read_csv(root_path + '/Data/BCH_USDT.csv', index_col=0, parse_dates=True)


btc_df['symbol'] = ['btc_df']*len(btc_df)
eth_df['symbol'] = ['eth_df']*len(eth_df)
xrp_df['symbol'] = ['xrp_df']*len(xrp_df)
ltc_df['symbol'] = ['ltc_df']*len(ltc_df)
bch_df['symbol'] = ['bch_df']*len(bch_df)

cryptos_df = pd.concat([btc_df, eth_df, xrp_df, ltc_df, bch_df], axis=0)
cryptos_df.to_csv(data_path + "cryptos_hf_0804.csv")


