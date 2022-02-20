import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from auxiliary_functions import rescale_sopr, rescale_extension, calculate_extension

pd.set_option("display.max_columns", None)

API_KEY = '257eUgVAg2YcQU6j4GgzyqB9rbj'
urls = [#'https://api.glassnode.com/v1/metrics/market/price_usd_close',
        'https://api.glassnode.com/v1/metrics/market/price_usd_ohlc',
        'https://api.glassnode.com/v1/metrics/indicators/sopr']


def plot_figure(y_range, x_data, y_data, title):

    plt.title(title)
    colour_map = plt.cm.get_cmap('RdYlGn').reversed()
    plt.scatter(x_data, y_data, c=y_data, cmap=colour_map)
    plt.ylim(y_range[0], y_range[1])
    plt.colorbar()
    plt.xticks(rotation=50)
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.show()
    plt.close()

# def plot_price_data(df):
#     candles = go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'])
#     sma_20_week = go.Scatter(x=df.index, y=df['close'].rolling(window=140).mean(), mode='lines', line={'color': '#1900ff'})
#     sma_50_week = go.Scatter(x=df.index, y=df['close'].rolling(window=350).mean(), mode='lines', line={'color': '#ff006a'})
#     # ema_21_week =
#
#     fig = go.Figure(data=[candles, sma_20_week, sma_50_week])
#     fig.update_layout(
#         title=f'The Candlestick graph for BTC',
#         xaxis_title='Date',
#         yaxis_title=f'Price (USD)',
#         #xaxis_rangeslider_visible=False
#     )
#     fig.update_yaxes(type="log")
#     fig.show()

def import_data():

    data = []
    for url in urls:
        label = url.split('/')[-1]

        # make API request
        res = requests.get(url, params={'a': 'BTC', 'api_key': API_KEY})

        # convert to dataframe
        df = pd.read_json(res.text, convert_dates=['t'])
        df.set_index('t', inplace=True)
        df.index.rename('date', inplace=True)
        df.rename(columns={'v': label}, inplace=True)
        data.append(df)

    data_df = pd.concat(data, axis=1)

    # split open, close, high, low data into individual columns
    close = pd.Series([value['c'] for value in data_df['o']], index=data_df.index, name='close')
    open = pd.Series([value['o'] for value in data_df['o']], index=data_df.index, name='open')
    high = pd.Series([value['h'] for value in data_df['o']], index=data_df.index, name='high')
    low = pd.Series([value['l'] for value in data_df['o']], index=data_df.index, name='low')
    data_df = data_df.join(pd.concat([open, close, high, low], axis=1))
    data_df.drop('o', axis=1, inplace=True)

    # get fear and greed index data
    res = requests.get('https://api.alternative.me/fng/?limit=0')

    fng_values = [int(value['value']) for value in json.loads(res.text)['data']]
    fng_dates = pd.Series([int(value['timestamp']) for value in json.loads(res.text)['data']])
    fng_dates = fng_dates.apply(datetime.fromtimestamp).apply(datetime.date)
    df = pd.DataFrame(fng_values, index=fng_dates, columns=["fear_and_greed"])

    # join all data into one dataframe
    data_df = data_df.join(df)

    return data_df.drop(data_df.index[data_df.index < datetime(2012, 1, 1)])




def main():

    df = import_data()

    df = calculate_extension(df)
    print(df)


    #print(df)

    # process sopr values
    df['sopr'] = preprocessing.StandardScaler().fit_transform(df['sopr'].values.reshape(-1, 1))[:, 0]
    df['sopr'] = df['sopr'].rolling(60).mean().apply(rescale_sopr)
    df['sopr'] = preprocessing.MinMaxScaler().fit_transform(df['sopr'].values.reshape(-1, 1))[:, 0]

    # process extension values
    df['extension'] = preprocessing.StandardScaler().fit_transform(df['extension'].values.reshape(-1, 1))[:, 0]
    df['extension'] = df['extension'].apply(rescale_extension)
    df['extension'] = preprocessing.MinMaxScaler().fit_transform(df['extension'].values.reshape(-1, 1))[:, 0]


    # plot_figure([accumulation_score.min(), accumulation_score.max()], accumulation_score.dropna().index, accumulation_score.dropna(),
    #             "Accumulation Score")


    plt.xlim(datetime(2019,1,1), datetime.today() + timedelta(days=30))
    plt.ylim(0.5*df[df.index > datetime(2019,1,1)]['close'].min(), 1.25*df[df.index > datetime(2019,1,1)]['close'].max())
    accumulation_score = (df['sopr'] + df['extension']) / 2
    plt.title("Long Term Bitcoin Accumulation Zones")
    colour_map = plt.cm.get_cmap('RdYlGn').reversed()
    plt.scatter(df.index, df['close'], c=accumulation_score, cmap=colour_map)
    plt.plot(df.index, df['sma20'], color='purple')
    plt.plot(df.index, df['ema21'], color='black')
    plt.colorbar()
    plt.xticks(rotation=50)
    plt.tight_layout()
    #plt.gca().set_yscale('log')
    plt.savefig("BTC Long Term Accumulation Zones.png")
    plt.show()
    #print(np.percentile(df['sopr'].dropna(), 98), np.percentile(df['sopr'].dropna(), 2))
    # plt.scatter(df.index, df['close'], color='orange')
    #
    #
    # # SHORT TERM ACCUMULATION
    # short_term_accumulation = df[df['sopr'].rolling(60).mean() < 1.00]
    # plt.scatter(short_term_accumulation.index, short_term_accumulation['close'], color='green')

    # MEDIUM TERM ACCUMULATION

    # LONG TERM ACCUMULATION

    # SHORT TERM DISTRIBUTION

    # MEDIUM TERM DISTRIBUTION

    # LONG TERM DISTRIBUTION

    # long_term_distribution = df[df['sopr'].rolling(30).mean() > 1.01]
    # plt.scatter(long_term_distribution.index, long_term_distribution['close'], color='red')
    # plt.show()
    #plt.close()


    # # plot each metric
    # plot_figure([df['sopr'].min(), df['sopr'].max()], df['sopr'].dropna().index, df['sopr'].dropna(),
    #             "Spent Output Ratio")
    #
    # plot_figure([df['nupl'].min(), df['nupl'].max()], df['nupl'].dropna().index, df['nupl'].dropna(),
    #             "Net Unrealized Profit Loss")
    # return
    #
    # plot_figure([0, 100], df['fear_and_greed'].dropna().index, df['fear_and_greed'].dropna(),
    #             "Fear and Greed Index")



if __name__ == '__main__':
    main()
    print('done')
