import pandas as pd
from pandas.tseries.offsets import DateOffset

df = pd.read_pickle('eth_hist.pkl')

# create missing rows
timestamps = pd.date_range(start=df['starttime'].dt.date.min(), end=df['starttime'].dt.date.max(), freq='10min')
left_df = pd.DataFrame(pd.Series(timestamps, name='starttime').dt.tz_localize('UTC'))
df = left_df.merge(df, how='left', on='starttime')
# for missing rows, set volume to 0, use last close value for prices
df['volume'] = df['volume'].fillna(0)
df['close'] = df['close'].fillna(method='ffill')
df['open'] = df['open'].fillna(df['close'])
df['low'] = df['low'].fillna(df['close'])
df['high'] = df['high'].fillna(df['close'])
print(len(df))

# simple calcs
df['delta'] = df['close'] - df['open']
df['spread'] = df['high'] - df['low']
df['percent delta'] = df['delta'] / df['open']
df['percent spread'] = df['spread'] / df['open']

def compile_set(df):
    df2 = df.copy()
    df2.index = range(len(df2.index))
    df2['norm close'] = (df2['close'] - df2['open']) / df2['open']
    total_volume = df2['volume']
    for i in range(143):
        df3 = df2.copy()[['close', 'volume']]
        df3.index = df3.index - i - 1
        cols = ['norm ' + x + ' ' + str(i+1) for x in df3.columns]
        df3.columns = cols
        df2 = df2.join(df3, how='left')
        total_volume = total_volume + df2['norm volume ' + str(i+1)]
        df2['norm close ' + str(i+1)] = (df2['norm close ' + str(i+1)] - df2['open']) / df2['open']
    # normalize volume
    df2['norm volume'] = df2['volume'] / total_volume
    for i in range(143):
        df2['norm volume ' + str(i+1)] = df2['norm volume ' + str(i+1)] / total_volume
    df2['total volume'] = total_volume
    # integral calculations
    df2['norm close integral'] = df2[[c for c in df2.columns if 'norm close' in c]].sum(axis=1)
    # score column
    scores = df2.copy()[['close', 'volume']]
    scores['include'] = scores['volume'] > 0
    scores.index = scores.index - 144
    df2['next point norm close'] = scores['close']
    df2['next point norm close'] = (df2['next point norm close'] - df2['open']) / df2['open']
    df2['include'] = scores['include']
    # drop tail
    df2 = df2.iloc[:-144]
    # drop 0 volume scores
    df2 = df2[df2['include']]
    df2 = df2[df2['total volume'] > 0]
    # drop columns
    df2 = df2.drop(['starttime', 'open', 'close', 'volume', 'low', 'high', 'delta', 'spread', 'percent delta', 'percent spread', 'include'], axis=1)
    # sum_df = pd.DataFrame()
    # for i in range(len(df) - 144):
    #     if df['volume'].iloc[i+144] > 0:
    #         # slice to day
    #         df_day = df.iloc[i:i + 144]
    #         # save start price
    #         p0 = df_day['open'].iloc[0]
    #         # normalize prices vs start price
    #         df_day['norm open'] = (df_day['open'] - p0) / p0
    #         df_day['norm close'] = (df_day['close'] - p0) / p0
    #         # normalize volume over period
    #         total_volume = df_day['volume'].sum()
    #         df_day['norm volume'] = df_day['volume'] / total_volume
    #         # calculate price integral
    #         p_int = df_day['norm close'].sum()
    #         # calculate price derivative
    #         p_der = df_day['norm close'].iloc[-1] - df_day['norm close'].iloc[-2]
    #         # pivot df
    #         df_day = df_day.set_index('starttime')
    #         df_day.index = range(len(df_day.index))
    #         df_day = df_day[['norm volume', 'norm close']].stack().reset_index() # could include 'norm volume', 'percent spread', 'norm close'
    #         df_day['header'] = df_day['level_1'].astype(str) + ' ' + df_day['level_0'].astype(str)
    #         df_day = df_day.drop(['level_0', 'level_1'], axis=1).set_index('header').transpose()
    #         df_day.index = [i]
    #         # include calculations
    #         df_day['total volume'] = total_volume
    #         df_day['norm close integral'] = p_int
    #         df_day['norm close derivative'] = p_der
    #         # score column
    #         df_day['next point percent delta'] = df['percent delta'].iloc[i+144]
    #         sum_df = sum_df.append(df_day)
    return df2

# create training data
train = compile_set(df[df['starttime'] < '2017-05-01'])
print(len(train), len(train.columns))
train.to_pickle('train_rolling.pkl')

# create test data
test = compile_set(df[df['starttime'] >= '2017-05-01'])
print(len(test), len(test.columns))
test.to_pickle('test_rolling.pkl')
