from dynamica_data_loader.DynamicaDataLoader import DynamicaDataLoader
from Code.tokens_cryptos import blackchain_token
import pandas as pd
import datetime as dt
import os

loader = DynamicaDataLoader(**blackchain_token)

top_coins = ['btc_usdt', 'eth_usdt', 'xrp_usdt', 'ltc_usdt', 'xmr_usdt']
nyc_tz = dt.timezone(dt.timedelta(hours=-5))

root_path = os.getcwd()

available_data = loader.get_available()

print(available_data)

columns_name = ['timestamp', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage']

btc_data = loader.get_data("price", "2015-12-31", "2018-07-01", query=top_coins[0])
eth_data = loader.get_data("price", "2015-12-31", "2018-07-01", query=top_coins[1])
xrp_data = loader.get_data("price", "2015-12-31", "2018-07-01", query=top_coins[2])
ltc_data = loader.get_data("price", "2015-12-31", "2018-07-01", query=top_coins[4])
xmr_data = loader.get_data("price", "2015-12-31", "2018-07-01", query=top_coins[3])


def form_df(coin_data):
    # coin_data = btc_data
    df = pd.DataFrame(data=coin_data, columns=columns_name)
    # df['time'] = [dt.datetime.fromtimestamp(float(timestamp), tz=nyc_tz) for timestamp in df['timestamp']]
    df.set_index(keys='timestamp', drop=True, inplace=True)
    return df


btc_df = form_df(btc_data)
eth_df = form_df(eth_data)
xrp_df = form_df(xrp_data)
xmr_df = form_df(xmr_data)
ltc_df = form_df(ltc_data)

btc_df.to_csv(root_path + '/Data/BTC_USDT.csv')
eth_df.to_csv(root_path + '/Data/ETH_USDT.csv')
xrp_df.to_csv(root_path + '/Data/XRP_USDT.csv')
xmr_df.to_csv(root_path + '/Data/BCH_USDT.csv')
ltc_df.to_csv(root_path + '/Data/LTC_USDT.csv')


# Combine constitutes
