import pandas as pd
import requests
import json
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math
from sklearn import preprocessing

def import_data(urls, api_key):

    data = []
    for url in urls:
        label = url.split('/')[-1]

        # make API request
        res = requests.get(url, params={'a': 'BTC', 'api_key': api_key})

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

    #return data_df.drop(data_df.index[data_df.index < datetime(2012, 1, 1)])
    return data_df

def calculate_extension(df):
    # extension from 20 week sma/21 week ema
    df['sma20'] = df['close'].rolling(140).mean()
    df['ema21'] = df['close'].ewm(span=147, adjust=False).mean()

    above = df[(df['close'] > df['sma20']) & (df['close'] > df['ema21'])]
    below = df[(df['close'] < df['sma20']) & (df['close'] < df['ema21'])]
    neutral = df[(~df.index.isin(above.index)) & (~df.index.isin(below.index))]

    above = (above['close'] / above[['sma20', 'ema21']].min(axis=1)).apply(lambda value: value - 1.0)
    below = (below['close'] / below[['sma20', 'ema21']].max(axis=1)).apply(lambda value: value - 1.0)
    neutral = neutral['close'].apply(lambda value: 0)

    above.rename('extension', inplace=True)
    below.rename('extension', inplace=True)
    neutral.rename('extension', inplace=True)

    return pd.concat([above, below, neutral], axis=0)


# RESCALING FUNCTIONS
def rescale_sopr(x):
    if x > 0.6:
        x = 0.6
    elif x < -0.8:
        x = -0.8
    return x


def rescale_extension(x):
    if x > 2:
        x = 2
    elif x < -1.1:
        x = 1.1
    return x

def rescale_fair_value(fair_value_extension):

    limit = pd.Series(-3*np.log(np.array(fair_value_extension.index)) + 28, index=fair_value_extension.index)
    #print(limit, fair_value_extension)

    for i in fair_value_extension.index:
        if fair_value_extension.loc[i] > limit.loc[i]:
            fair_value_extension.loc[i] = limit.loc[i]

    fair_value_extension = 1 - (limit - fair_value_extension).apply(lambda x: 2 * abs(x)) / (limit + fair_value_extension)

    return fair_value_extension



# LOGARITHMIC REGRESSION FUNCTIONS
def objective(x, a, b, c, d):
    return a * np.log(b * x + c) + d


def logarithmic_regression(y_data, index=None, df_index=None):

    x_values = [x + 1 for x in range(len(y_data))] if index is None else [i+1 for i in index]
    y_values = np.log(y_data)

    popt, _ = curve_fit(objective, x_values, y_values, p0=(5, 0.5, 70, -15), bounds=((0, 0, 0, -1000), (100, 100, 10000, 1000)))
    #print(popt)
    if df_index is None:
        return np.exp(objective(np.array(x_values), popt[0], popt[1], popt[2], popt[3]))

    return np.exp(objective(np.array([x + 1 for x in range(len(df_index))]), popt[0], popt[1], popt[2], popt[3]))



def fair_value_regression(df):
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots()
    plt.title("Bitcoin Fair Value Logarithmic Regression Band")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')

    # plot BTC price
    plt.semilogy(df.index, df['close'], color='white', lw=0.5)

    # plot regression bands
    fair_value = logarithmic_regression(df['close'])

    for i in [-0.15, 0, 0.15]:

        if i == 0:
            ax1.plot(df.index, fair_value * np.exp(i), color='yellow', lw=0.5)
        else:
            ax1.plot(df.index, fair_value * np.exp(i), color='yellow', alpha=0.5, lw=0.4)

        if i != -0.15:
            plt.fill_between(df.index, fair_value * np.exp(i - 0.15), fair_value * np.exp(i), alpha=0.4, color='yellow')

    plt.xticks(rotation=50)
    plt.tight_layout()

    plt.savefig('figs//fair_value_regression_band.png', dpi=300)
    #plt.show()
    plt.close()

    return fair_value


def regression_bands(df):
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots()
    plt.title("Bitcoin Logarithmic Regression Bands")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')

    # plot BTC price
    plt.semilogy(df.index, df['close'], color='white', lw=0.5)

    # plot regression bands
    fair_value = logarithmic_regression(df['close'])

    for i in [-0.5, 0, 0.5, 1, 1.5, 2, 2.5]:
        ax1.plot(df.index, fair_value * np.exp(i), lw=0.5)
        plt.fill_between(df.index, fair_value * np.exp(i - 0.5), fair_value * np.exp(i), alpha=0.4)

    plt.xticks(rotation=50)
    plt.tight_layout()

    plt.savefig('figs//logarithmic_regression_bands.png', dpi=300)
    plt.close()


def fair_value_extension(df, fair_value):
    fv_extension = (df['close'] / fair_value).apply(lambda value: value - 1.0)
    fv_extension = pd.Series(list(fv_extension), index=[x + 1 for x in range(len(fair_value))])
    fv_extension = rescale_fair_value(fv_extension)
    fv_extension = preprocessing.MinMaxScaler().fit_transform(np.array(fv_extension).reshape(-1, 1))[:, 0]

    return fv_extension

def sma_200_extension(df, sma_200):
    return (df['close']/sma_200).apply(lambda value: value - 1)

def long_term_risk_metric(df, accumulation_score):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(14)

    ax1.set_title("Bitcoin Long Term Accumulation Zones")
    ax1.set_xlabel('Date')
    ax1.set_yscale('log')
    ax1.set_ylabel('Price')

    ax2.set_title('Long Term Risk Metric')
    ax2.set_xlabel('Date')

    colour_map = plt.cm.get_cmap('RdYlGn').reversed()
    ax1.scatter(df.index, df['close'], c=accumulation_score, cmap=colour_map)

    for i in np.linspace(0.1, 1.0, 10):

        if round(i, 2) <= 0.3:
            colour = 'green'
        elif round(i, 2) >= 0.7:
            colour = 'red'
        else:
            colour = 'yellow'

        ax2.fill_between(df.index, i - 0.1, i, alpha=abs(i - 0.5), color=colour)

    ax2.plot(df.index, accumulation_score, color='white', alpha=0.5)
    plt.savefig('figs//Bitcoin Long Term Accumulation Zones')
    plt.show()
    plt.close()

