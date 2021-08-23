import pandas as pd
import datetime
import json
import requests

monedas = ["LTCUSDT", "BCHUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT"]

url = "https://api.binance.com/api/v3/klines"

interval = "1d"

headers = ["datetime",
           "open",
           "high",
           "low",
           "close",
           "volume",
           "Close time",
           "Quote asset volume",
           "trades",
           "Taker buy base asset volume",
           "Taker buy quote asset volume",
           "Ignore"]

daysInterval = 200
limit = 1000
for moneda in monedas:
    startTime = datetime.datetime(2018, 1, 1)
    nextTime = startTime + datetime.timedelta(daysInterval)
    endTime = datetime.datetime(2021, 7, 31)

    df = pd.DataFrame(columns=headers)

    while(nextTime < endTime):
        startTimeReq = str(int(startTime.timestamp() * 1000))
        nextTimeReq = str(int(nextTime.timestamp() * 1000))
        request_params = {
            "symbol": moneda,
            "interval": interval,
            "startTime": startTimeReq,
            "endTime": nextTimeReq,
            "limit": limit
        }
        tmpDf = pd.DataFrame(json.loads(requests.get(url, params= request_params).text), columns=headers)
        df = pd.concat([df, tmpDf])

        startTime = nextTime + datetime.timedelta(1)
        nextTime = startTime + datetime.timedelta(daysInterval)

    df = df.loc[:, ["datetime", "open", "high", "low", "close", "volume", "trades"]]
    df.index = [datetime.datetime.fromtimestamp(x / 1000) for x in df.datetime]
    print("Moneda: " + moneda + ", con un tamano de: " + str(df.shape))
    df.to_csv(moneda + '.csv', encoding='utf-8', index=True)

'''
startTime = str(int(datetime.datetime(2020, 7, 1).timestamp() * 1000))
endTime = str(int(datetime.datetime(2021, 7, 30).timestamp() * 1000))


df = pd.DataFrame(json.loads(requests.get(url, params= request_params).text), columns=headers)
df = df.loc[:, ["datetime", "open", "high", "low", "close", "volume", "trades"]]

df.index = [datetime.datetime.fromtimestamp(x / 1000) for x in df.datetime]
'''
