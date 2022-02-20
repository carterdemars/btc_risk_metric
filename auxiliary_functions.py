import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

def calculate_extension(df):

    # extension from 20 week sma
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

    return pd.concat([df, pd.concat([above, below, neutral], axis=0)], axis=1)

# RESCALING FUNCTIONS
def rescale_sopr(x):
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return x


def rescale_extension(x):
    if x > 2:
        x = 2
    return x