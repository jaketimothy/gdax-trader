import pandas as pd
from pandas.tseries.offsets import DateOffset

df = pd.read_pickle('eth_hist.pkl')
print(len(df))

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

# simple calcs
df['delta'] = df['close'] - df['open']
df['spread'] = df['high'] - df['low']
df['percent delta'] = df['delta'] / df['open']
df['percent spread'] = df['spread'] / df['open']

def compile_set(df, days):
    sum_df = pd.DataFrame()
    for i in range(len(days) - 1):
        # get
        df_day = df[(df['starttime'] >= days[i]) & (df['starttime'] < days[i+1])]
        p0 = df_day['open'].iloc[0]
        df_day['norm open'] = (df_day['open'] - p0) / p0
        df_day['norm close'] = (df_day['close'] - p0) / p0
        df_day = df_day.set_index('starttime')
        df_day.index = range(len(df_day.index))
        # idealy: ['volume', 'percent spread', 'norm close']
        df_day = df_day[['volume', 'norm close']].stack().reset_index()
        df_day['header'] = df_day['level_1'].astype(str) + ' ' + df_day['level_0'].astype(str)
        df_day = df_day.drop(['level_0', 'level_1'], axis=1).set_index('header').transpose()
        next_day = df[df['starttime'] < days[i+1] + DateOffset(days=1)]['starttime'].max()
        next_day_start = df[df['starttime'] >= days[i+1]]['starttime'].min()
        p1 = df[df['starttime'] == next_day_start]['open'].iloc[0]
        df_day['next day norm close'] = (df[df['starttime'] == next_day]['close'].iloc[0] - p1) / p1
        df_day.index = [days[i]]
        sum_df = sum_df.append(df_day)
    return sum_df

# create training data
days = pd.date_range(start=df['starttime'].dt.date.min(), end='2017-05-01')
train = compile_set(df, days)
print(len(train), len(train.columns))
train.to_pickle('train.pkl')

# create test data
days = pd.date_range(start='2017-05-01', end=df['starttime'].dt.date.max())
test = compile_set(df, days)
print(len(test), len(test.columns))
test.to_pickle('test.pkl')
