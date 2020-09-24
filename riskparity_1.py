from datetime import datetime, date
import math
import numpy as np
import time
import sys
import requests
import matplotlib as plt

# if len(sys.argv) == 1:
#     symbols = ['UPRO', 'TMF']
# else:
#     symbols = sys.argv[1].split(',')
#     for i in range(len(symbols)):
#         symbols[i] = symbols[i].strip().upper()

symbols = [#'BND',
           # 'TMF',
           # 'TLT',
           # 'EDV',
           # 'VOO',
           # 'QQQ',
           # 'TQQQ',
           # 'NTSX',
           # 'SWAN',
           # 'ARKG',
           'ARKK',
           # 'ARKF',
           'ARKW'
           ]

# 20 days set: TQQQ, TNA, UPRO, TMF
# 60 dasy set: ARKK, TNA, UPRO, SWAN

window_size = 60 #business days can test with 60 days

num_trading_days_per_year = 252
date_format = "%Y-%m-%d"
end_timestamp = int(time.time())
start_timestamp = int(end_timestamp - (7/5 * (window_size + 1) + 4) * 86400) # a month ago

def get_Vol_Ret(symbol):

    download_url = "https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history".format(symbol, start_timestamp, end_timestamp)
    # raw data
    lines = requests.get(download_url, cookies={'B': 'chjes25epq9b6&b=3&s=18'}).text.strip().split('\n')

    # for debug, will print error if assert fails    
    assert lines[0].split(',')[0] == 'Date'
    assert lines[0].split(',')[4] == 'Close'

    # get close price from latest to earliest
    prices = []
    for line in lines[1:]:
        prices.append(float(line.split(',')[4]))
    prices.reverse()
    # print(lines)
    # print(prices)

    # calculate log return
    volatilities_in_window = []
    for i in range(window_size):
        volatilities_in_window.append(math.log(prices[i] / prices[i+1])) 
    
    # check most recent trading date is close
    most_recent_date = datetime.strptime(lines[-1].split(',')[0], date_format).date()
    assert (date.today() - most_recent_date).days <= 4, "today is {}, most recent trading day is {}".format(date.today(), most_recent_date)

    # return the vol and return in the window
    return np.std(volatilities_in_window, ddof = 1) * np.sqrt(num_trading_days_per_year), prices[0] / prices[window_size] - 1.0


# calculate weights, apply to 3 and above as well?
volatilities = []
returns = []
sum_inverse_vol = 0.0

for symbol in symbols:
    vol, ret = get_Vol_Ret(symbol)
    sum_inverse_vol += 1 / vol
    volatilities.append(vol)
    returns.append(ret)    
weights = [100 / (vol * sum_inverse_vol) for vol in volatilities]
# print(sum_inverse_vol)
# print(volatilities)
# print(returns)
# print(weights)

print ("Portfolio: {}, as of {} (with window size of {} days)".format(str(symbols), date.today().strftime('%Y-%m-%d'), window_size))
for i in range(len(symbols)):
    print ('{} weight: {:.2f}% (anualized volatility: {:.2f}%, return: {:.2f}%)'.format(symbols[i], float(weights[i]), float(volatilities[i] * 100), float(returns[i] * 100)))









