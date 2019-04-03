import matplotlib as mpl

mpl.use('TKAgg')
from Code.RCVJ_RealizedVolatility.PLT_Funcs import *
from Code.RCVJ_DataOperation.IO_Funcs import read_dyos_clean, lower_freq

coin_names = ['BTC', 'ETH', 'XRP', 'LTC']
freq = '5min'  # For now, doesn't support different freq on this plot


def plot_trading_volume(coin_name):
    logreturn, close_price = read_dyos_clean(coin_name)
    volume = close_price['volume']
    plt.figure(figsize=(15,6))
    plt.plot(volume.index, volume.values)
    plt.ylabel('Volume', fontsize=16)
    plt.xlabel('Date', fontsize=16)
    plt.title(f'Trading Volume of {coin_name}')
    plt.tight_layout()
    plt.savefig(outplot_dir + f'TradingVolume{coin_name}.png', dpi=300)


for coin_name in coin_names:
    plot_trading_volume(coin_name)