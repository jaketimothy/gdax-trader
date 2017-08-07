from gdax import PublicClient
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

dates = pd.date_range(start='2016-06-01', end='2017-08-01')
hist_cols = [ 'starttime', 'low', 'high', 'open', 'close', 'volume' ]

def pull_save_hist(granularity=10):
    # granularity in minutes
    public_client = PublicClient()
    rates_list = []
    for i in range(len(dates) - 1):
        # gets candles per day
        rates_list.extend(public_client.get_product_historic_rates('ETH-USD', start=dates[i], end=dates[i+1], granularity=granularity*60))
    rates_list = [x for x in rates_list if not isinstance(x, basestring)] # remove garbage
    eth_hist = pd.DataFrame(rates_list, columns=hist_cols)
    eth_hist['starttime'] = eth_hist['starttime'].apply(lambda x: datetime.fromtimestamp(x, pytz.utc))
    eth_hist = eth_hist.sort_values(by='starttime')

    plt.figure()
    eth_hist.plot(x='starttime', y='open')
    plt.show()

    eth_hist.to_pickle('eth_hist_' + str(granularity) + '.pkl')
