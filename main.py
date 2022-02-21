import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from scipy.optimize import curve_fit
from sklearn import preprocessing
from auxiliary_functions import rescale_sopr, rescale_extension, calculate_extension, regression_bands, \
    fair_value_regression, fair_value_extension, import_data, rescale_fair_value, long_term_risk_metric


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






def main():

    df = import_data(urls, API_KEY)

    df = calculate_extension(df)


    # process sopr values
    df['sopr'] = preprocessing.StandardScaler().fit_transform(df['sopr'].values.reshape(-1, 1))[:, 0]
    df['sopr'] = df['sopr'].rolling(50).mean().apply(rescale_sopr)
    df['sopr'] = preprocessing.MinMaxScaler().fit_transform(df['sopr'].values.reshape(-1, 1))[:, 0]

    # process extension values
    df['extension'] = preprocessing.StandardScaler().fit_transform(df['extension'].values.reshape(-1, 1))[:, 0]
    df['extension'] = df['extension'].apply(rescale_extension)
    df['extension'] = preprocessing.MinMaxScaler().fit_transform(df['extension'].values.reshape(-1, 1))[:, 0]

    # process fear and green index values
    df['fear_and_greed'] = df['fear_and_greed'].apply(lambda x: x/100.0).ewm(span=50, adjust=False).mean()

    # plot_figure([accumulation_score.min(), accumulation_score.max()], accumulation_score.dropna().index, accumulation_score.dropna(),
    #             "Accumulation Score")


    # LONG TERM BTC OUTLOOK (1 YEAR +)

    regression_bands(df)
    fair_value = pd.Series(fair_value_regression(df), index=df.index)
    fve = fair_value_extension(df, fair_value)

    #
    # # OTHER PLOTTING
    #
    # fig, ax1 = plt.subplots()
    #
    # ax2 = ax1.twinx()
    #
    # ax1.set_xlabel('Date')
    # ax1.set_ylabel('Price')
    # ax2.set_ylabel('Accumulation Score')
    #plt.xlim(datetime(2019,1,1), datetime.today() + timedelta(days=30))
    #plt.ylim(0.5*df[df.index > datetime(2019,1,1)]['close'].min(), 1.25*df[df.index > datetime(2019,1,1)]['close'].max())
    accumulation_score = (df['sopr'] + df['extension'] + df['fear_and_greed'] + fve) / 4.0
    accumulation_score[df['fear_and_greed'].isna()] = (df['sopr'] + df['extension'] + fve) / 3.0
    long_term_risk_metric(df, accumulation_score)



    # #accumulation_score = df['fear_and_greed']
    # plt.title("Long Term Bitcoin Accumulation Zones")
    # colour_map = plt.cm.get_cmap('RdYlGn').reversed()
    # ax1.scatter(df.index, df['close'], c=accumulation_score, cmap=colour_map)
    # ax2.plot(accumulation_score)
    # #plt.plot(df.index, df['sma20'], color='purple')
    # #plt.plot(df.index, df['ema21'], color='black')
    # #plt.plot()
    # #fig.colorbar()
    # # ax1.plot(df['close'].rolling(200*7).mean(), color='black')
    # # ax2.plot(df['close']/df['close'].rolling(200*7).mean())
    # #ax1.plot(df.index, fair_value)
    # plt.xticks(rotation=50)
    # plt.tight_layout()
    # ax1.set_yscale('log')
    # plt.savefig("figs//BTC Long Term Accumulation Zones.png")
    # plt.show()
    # return
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
