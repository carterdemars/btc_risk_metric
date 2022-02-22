from main import main
from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy
import pandas as pd
from datetime import date, datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.2e" % x)
output = "{:.2e}"


# prompt user for how much bitcoin they own and how much cash they want to spend
balance = 5000 # value in USD
start_date = date(2021,1,1)
trade_value = 0 # value in BTC

score, price = main()
score = score.loc[score.index > pd.to_datetime(start_date)]
price = price.loc[price.index > pd.to_datetime(start_date)]


trade_value_over_time = []
balance_over_time = []
messages = []


for date, risk in score.iteritems():


    btc_price = price.loc[pd.to_datetime(date)]['close']

    #print(btc_price)
    #print(date, risk, btc_price)

    if risk < 0.3 and balance > 0.01:

        if risk < 0.1:
            buy_amt_usd = 0.25 * balance
        elif risk < 0.2:
            buy_amt_usd = 0.05 * balance
        else:
            buy_amt_usd = 0.015 * balance

        buy_amt_btc = buy_amt_usd/btc_price
        messages.append(f"B {output.format(buy_amt_btc)} for {output.format(buy_amt_usd)} at {output.format(btc_price)}")

        trade_value = trade_value + buy_amt_btc
        balance = balance - buy_amt_usd
        trade_value_over_time.append(trade_value)
        balance_over_time.append(balance)

        print("Buy", round(buy_amt_usd, 2), round(btc_price, 2), round(buy_amt_btc, 2), round(trade_value, 2),round(balance))
        # if risk < 0.1:
        #     trade_value += buy_amt_btc
        #     balance -= buy_amt_usd
        # elif risk < 0.2:
        #     trade_value += buy_amt_btc
        #     balance -= buy_amt_usd
        # else:
        #     trade_value += buy_amt_btc
        #     balance -= buy_amt_usd

    elif risk > 0.7 and trade_value > 0.01:

        if risk > 0.9:
            sell_amt_btc = 0.25 * trade_value
        elif risk > 0.8:
            sell_amt_btc = 0.05*trade_value
        else:
            sell_amt_btc = 0.02*trade_value

        sell_amt_usd = sell_amt_btc*btc_price
        messages.append(f"S {output.format(sell_amt_btc)} for {output.format(sell_amt_usd)} at {output.format(btc_price)}")

        trade_value = trade_value - sell_amt_btc
        balance = balance + sell_amt_usd
        trade_value_over_time.append(trade_value)
        balance_over_time.append(balance)

        print("Sell", round(sell_amt_usd,2), round(btc_price,2), round(sell_amt_btc,2), round(trade_value,2), round(balance))
        # if risk > 0.9:
        #     balance += sell_amt_usd
        #     trade_value -= sell_amt_btc
        # elif risk > 0.8:
        #     balance += sell_amt_usd
        #     trade_value -= sell_amt_btc
        # else:
        #     balance += sell_amt_usd
        #     trade_value -= sell_amt_btc
    else:
        messages.append("No action")
        trade_value_over_time.append(trade_value)
        balance_over_time.append(balance)
        print("Hold", 0, round(btc_price,2), 0, round(trade_value,2), round(balance))

# print(trade_value_over_time)
# print(balance_over_time)

bitcoin_owned = pd.Series(trade_value_over_time, index=price.index)
trade_value_over_time = pd.Series(list(trade_value_over_time*price['close']), index=price.index, name='trade_value')
balance_over_time = pd.Series(balance_over_time, index=price.index, name='account_balance')
messages = pd.Series(messages, index=price.index)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.set_figwidth(15)
fig.set_figheight(10)
ax1.set_title('Bitcoin Owned')
#ax1.set_yscale('log')
ax1.set_ylim(0, 1.5* bitcoin_owned.max())
ax1.plot(price.index, bitcoin_owned, color='orange')
plt.xticks(rotation=50)
ax2.set_yscale('log')
ax2.set_ylim(1, 1.25*max(trade_value_over_time.max(), balance_over_time.max()))
ax2.set_title('Liquid Value')
ax2.plot(price.index, trade_value_over_time, color='purple')
ax2.plot(price.index, balance_over_time, color='yellow')
plt.xticks(rotation=50)
ax3.set_yscale('log')
ax3.set_title('Total Value')
total_value = trade_value_over_time + balance_over_time
ax3.set_ylim(0.75* total_value.min(), 1.5*total_value.max())
ax3.plot(price.index, total_value, color='green')
plt.xticks(rotation=50)
plt.tight_layout()
plt.show()

#my_df = pd.concat([trade_value_over_time*price['close'], balance_over_time, messages], axis=1)
#print(my_df.round(-4))
#print(trade_value_over_time, balance_over_time)


# price.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
# indices = price.index[price.index < pd.to_datetime(date(2012,1,1))]
# # score.drop(indices, inplace=True)
# # price.drop(indices, inplace=True)
# #price = pd.DataFrame(data=[list(open), list(high), list(low), list(close)], index = close.index, columns=['Open', 'High', 'Low', 'Close'])
# print(score, price)
#
#
# class AccumulationStrategy(SignalStrategy):
#
#     def init(self):
#         super().init()
#         self.score, self.price = main()
#
#         buy_signal = pd.Series(self.score < 0.3).astype(int).diff().fillna(0)
#         buy_signal = buy_signal.replace(-1, 0)
#
#         entry_size = buy_signal*0.5
#
#         sell_signal = pd.Series(self.score > 0.7).astype(int).diff().fillna(0)
#         sell_signal = sell_signal.replace(1, 0)
#
#         exit_size = sell_signal * 0.5
#
#         self.set_signal(entry_size=entry_size, exit_portion=exit_size)
#
#
#
# bt = Backtest(pd.DataFrame(price), AccumulationStrategy,
#               cash=1000000, commission=.002,
#               exclusive_orders=True)
#
# output = bt.run()
# print(output)
# bt.plot()
# bt.show()

# def objective(params):
#     a, b, c, d, e, f = params
#
#     balance = 10000  # value in USD
#     trade_value = 0  # value in BTC
#
#     for date, risk in score.iteritems():
#
#
#         btc_price = price.loc[pd.to_datetime(date)]['close']
#
#         #print(btc_price)
#         #print(date, risk, btc_price)
#
#         if risk < 0.3 and balance > 0.01:
#
#             if risk < 0.1:
#                 buy_amt_usd = a * balance
#             elif risk < 0.2:
#                 buy_amt_usd = b *balance
#             else:
#                 buy_amt_usd = c * balance
#
#             buy_amt_btc = buy_amt_usd/btc_price
#             trade_value = trade_value + buy_amt_btc
#             balance = balance - buy_amt_usd
#
#
#             #print(date, round(buy_amt_usd, 2), round(btc_price, 2), round(buy_amt_btc, 2), round(trade_value, 2),round(balance))
#             # if risk < 0.1:
#             #     trade_value += buy_amt_btc
#             #     balance -= buy_amt_usd
#             # elif risk < 0.2:
#             #     trade_value += buy_amt_btc
#             #     balance -= buy_amt_usd
#             # else:
#             #     trade_value += buy_amt_btc
#             #     balance -= buy_amt_usd
#
#         elif risk > 0.7 and trade_value > 0.01:
#
#             if risk > 0.9:
#                 sell_amt_btc = d * trade_value
#             elif risk > 0.8:
#                 sell_amt_btc = e *trade_value
#             else:
#                 sell_amt_btc = f *trade_value
#
#             sell_amt_usd = sell_amt_btc*btc_price
#
#             trade_value = trade_value - sell_amt_btc
#             balance = balance + sell_amt_usd
#
#
#             #print(date, round(sell_amt_usd,2), round(btc_price,2), round(sell_amt_btc,2), round(trade_value,2), round(balance))
#             # if risk > 0.9:
#             #     balance += sell_amt_usd
#             #     trade_value -= sell_amt_btc
#             # elif risk > 0.8:
#             #     balance += sell_amt_usd
#             #     trade_value -= sell_amt_btc
#             # else:
#             #     balance += sell_amt_usd
#             #     trade_value -= sell_amt_btc
#
#     return -(balance + trade_value * btc_price)
#
# for year in [2020, 2021, 2022]:
#     score = score.loc[score.index > pd.to_datetime(date(year,1,1))]
#     price = price.loc[price.index > pd.to_datetime(date(year,1,1))]
#     x0 = np.array([0.1,0.05,0.01,0.1,0.05,0.01])
#     best = optimize.minimize(objective, x0, bounds=((0,1),(0,1),(0,1),(0,1),(0,1),(0,1)))
#     print(year, best['x'])

# a = 0.25
# b = 0.05
# c = 0.01
# d = 0.25
# e = 0.05
# f = 0.02