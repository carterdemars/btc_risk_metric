import pandas as pd
import requests
import json
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from scipy.optimize import curve_fit
from sklearn import preprocessing
from auxiliary_functions import rescale_sopr, rescale_extension, calculate_extension, regression_bands, \
    fair_value_regression, fair_value_extension, import_data, long_term_risk_metric, logarithmic_regression, \
    sma_200_extension, send_message


pd.set_option("display.max_columns", None)

API_KEY = '257eUgVAg2YcQU6j4GgzyqB9rbj'
urls = ['https://api.glassnode.com/v1/metrics/market/price_usd_ohlc',
        'https://api.glassnode.com/v1/metrics/indicators/sopr']


# def plot_figure(y_range, x_data, y_data, title):
#
#     plt.title(title)
#     colour_map = plt.cm.get_cmap('RdYlGn').reversed()
#     plt.scatter(x_data, y_data, c=y_data, cmap=colour_map)
#     plt.ylim(y_range[0], y_range[1])
#     plt.colorbar()
#     plt.xticks(rotation=50)
#     plt.tight_layout()
#     plt.savefig(f'{title}.png')
#     plt.show()
#     plt.close()

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
    df = pd.concat([df, calculate_extension(df)], axis=1)
    plt.style.use("dark_background")
    # print(date.today())
    # print(df)


    # process sopr values
    df['sopr'] = preprocessing.StandardScaler().fit_transform(df['sopr'].values.reshape(-1, 1))[:, 0]
    df['sopr'] = df['sopr'].rolling(30).mean().apply(rescale_sopr)
    df['sopr'] = preprocessing.MinMaxScaler().fit_transform(df['sopr'].values.reshape(-1, 1))[:, 0]
    #df.loc[df.index < pd.to_datetime(date(2012, 1, 1))]['sopr'] = np.nan
    # plt.plot(df['sopr'])
    # plt.show()

    # process extension values
    df['extension'] = preprocessing.StandardScaler().fit_transform(df['extension'].values.reshape(-1, 1))[:, 0]
    df['extension'] = df['extension'].apply(rescale_extension)
    df['extension'] = preprocessing.MinMaxScaler().fit_transform(df['extension'].values.reshape(-1, 1))[:, 0]
    # plt.plot(df['extension'])
    # plt.show()

    # process fear and green index values
    df['fear_and_greed'] = df['fear_and_greed'].apply(lambda x: x/100.0).ewm(span=50, adjust=False).mean()

    # plot_figure([accumulation_score.min(), accumulation_score.max()], accumulation_score.dropna().index, accumulation_score.dropna(),
    #             "Accumulation Score")

    # process 200w MA extension
    sma_200 = df['close'].rolling(1400).mean()
    sma_200 = sma_200_extension(df, sma_200)
    sma_200 = sma_200[sma_200 < 1.0]
    sma_200 = pd.Series(preprocessing.MinMaxScaler().fit_transform(abs(1/sma_200).values.reshape(-1, 1))[:, 0], index=sma_200.index)
    sma_200.drop(sma_200.index[sma_200 > 0.15], inplace=True)

    # LONG TERM BTC OUTLOOK (1 YEAR +)

    regression_bands(df)
    fair_value = pd.Series(fair_value_regression(df), index=df.index)
    fve = fair_value_extension(df, fair_value)
    # plt.plot(df.index, fve)
    # plt.show()
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

    # CALCULATE ACCUMULATION SCORE

    accumulation_score = (df['sopr'] + df['extension'] + df['fear_and_greed'] + fve) / 4.0
    accumulation_score[df['fear_and_greed'].isna()] = (df['sopr'] + df['extension'] + fve) / 3.0
    accumulation_score[df['sopr'].isna()] = (df['extension'] + fve) / 2.0
    accumulation_score[df.index.isin(sma_200.index)] -= sma_200

    long_term_risk_metric(df, accumulation_score)

    #print(df[accumulation_score > 0.8]['close'])

    temp_df = df.reset_index(drop=True)
    non_bubble_data = temp_df[accumulation_score.reset_index(drop=True) < 0.20]
    bubble_data = temp_df[accumulation_score.reset_index(drop=True) > 0.9]

    non_bubble_data = logarithmic_regression(non_bubble_data['close'], index=non_bubble_data.index, df_index=df.index)
    bubble_data = logarithmic_regression(bubble_data['close'], index=bubble_data.index, df_index=df.index)


    fig, ax1 = plt.subplots()
    plt.title("Bitcoin Fair Value Logarithmic Regression Bands")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')

    # plot BTC price
    # colour_map = plt.cm.get_cmap('RdYlGn').reversed()
    # ax1.scatter(df.index, df['close'], c=accumulation_score, cmap=colour_map)
    ax1.set_yscale('log')
    ax1.plot(df['close'], color='white', lw=0.5)
    #ax1.plot(sma_200, color='white')

    # ax1.plot(df.index, non_bubble_data, color='green', lw=0.5)
    # ax1.plot(df.index, bubble_data, color='red', lw=0.5)

    for i in [-0.2, 0, 0.2]:

        if i == 0:
            ax1.plot(df.index, non_bubble_data * np.exp(i), color='green', lw=0.5)
            ax1.plot(df.index, bubble_data * np.exp(i), color='red', lw=0.5)
        else:
            ax1.plot(df.index, non_bubble_data * np.exp(i), color='green', lw=0.25)
            ax1.plot(df.index, bubble_data * np.exp(i), color='red', lw=0.25)

        if i != -0.2:
            plt.fill_between(df.index, bubble_data * np.exp(i - 0.2), bubble_data * np.exp(i), alpha=0.4, color='red')
        plt.fill_between(df.index, non_bubble_data * np.exp(i - 0.2), non_bubble_data * np.exp(i), alpha=0.4, color='green')


    plt.xticks(rotation=50)
    plt.tight_layout()

    plt.savefig('figs//over_under_valuation_regression_bands.png', dpi=300)
    #plt.show()
    plt.close()

    todays_score = (accumulation_score.loc[accumulation_score.index == pd.to_datetime(date.today())]).values[0]
    todays_close = df.loc[df.index == pd.to_datetime(date.today())]['close'].values[0]

    # if todays_score < 0.3:
    #     send_message(round(todays_score, 3), round(todays_close))


    #print('done')
    return [accumulation_score.loc[accumulation_score.index >= pd.to_datetime(date(2012,1,1))],
            df.loc[df.index >= pd.to_datetime(date(2012,1,1))][['open', 'high', 'low', 'close']]]




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
