import matplotlib.pyplot as plt
from Code.GlobalParams import *
import pandas as pd

bitcoin_daily = pd.read_csv(data_dir + '/CryptosData/bitcoin_daily.csv', index_col=0, parse_dates=True, thousands=',')

volume = bitcoin_daily['Volume']

plt.figure(figsize=(15,7))
plt.bar(volume.index, volume.values)
plt.ylabel('Volume', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.tight_layout()
plt.savefig(outplot_dir + 'TradingVolume_Bitcoin_daily.png', dpi=300)
