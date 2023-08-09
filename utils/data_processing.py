import pandas as pd
from talib.abstract import SMA, EMA, RSI, MACD
import datetime as dt
from scipy import stats
import numpy as np

def load_data(path, start_date, end_date, indicators=True):
    
    # Load data from file
    df = pd.read_csv(path)
    df = df[["DateTime", "Close"]]
    df['date_ordinal'] = pd.to_datetime(df['DateTime']).map(dt.datetime.toordinal)
    df = df.set_index("DateTime")
    
    if indicators:
        # Add indicators
        for i in [5,10,20,60]:
            df['SMA'+ str(i)] = SMA(df['Close'])
            
        for i in [5,10,20,60]:
            df['EMA'+ str(i)] = EMA(df['Close'])
            
        for i in [5,10,20,60]:
            df['RSI'+ str(i)] = (RSI(df['Close']) - 50) / 50
            
        for i in [5,10,20,60]:
            df['MACD'+ str(i)], df['MACD'+ str(i) + "S"] , df['MACD'+ str(i) + "H"]  = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)    

        for i in [1,3,5,10,20,30]:
            df['FR'+ str(i)] = df['Close'].shift(-i) - df['Close']  
            df['FR'+ str(i) + "TF"] = df['FR'+ str(i)] > 0
            
        for i in [3,5,10,20,60]:
            df['SL'+ str(i) + 'CLOSE'] = get_slope_linregress(df['Close'], i)
            for j in [5,10,20,60]:
                df['SL'+ str(i) + 'SMA'+ str(j)] = get_slope_linregress(df['SMA'+ str(j)], i)
                df['SL'+ str(i) + 'RSI'+ str(j)] = get_slope_linregress(df['RSI'+ str(j)], i)
                
        for i in ['SMA5', 'SMA10', 'SMA20', 'SMA60']:
            for j in ['Close', 'SMA5', 'SMA10', 'SMA20', 'SMA60']:
                df['PD_'+ i + '_' + j] = get_percentage_difference(df[i], df[j])
                
        for i in ['RSI5', 'RSI10', 'RSI20', 'RSI60']:
            for j in ['RSI5', 'RSI10', 'RSI20', 'RSI60']:
                df['PD_'+ i + '_' + j] = get_percentage_difference(df[i], df[j])
                
        
    df = df[(df.index > start_date) & (df.index <= end_date)]   
    
    return df

def get_percentage_difference(item1, item2):
    return (item1 - item2) / item1

def get_slope_linregress(df, PERIOD):
    # print("len",len(df))
    key = df.name
    df = df.reset_index()
    df.DateTime =pd.to_datetime(df.DateTime)
    # df['date_ordinal'] = pd.to_datetime(df['DateTime']).map(dt.datetime.toordinal)
    df['date_ordinal'] = df['DateTime'].values.astype(np.int64) // 10 ** 10 // 6
    # print(df['date_ordinal'].head())
    df['slope'] = np.nan

    for i in range(len(df)):
        if i < (PERIOD-1):
            continue
        # print(df['date_ordinal'][i-(PERIOD-1):i+1], df[key][i-(PERIOD-1):i+1])
        slope, _, _, _, _ = stats.linregress(df['date_ordinal'][i-(PERIOD-1):i+1], df[key][i-(PERIOD-1):i+1])
        # print(slope)
        df.loc[i,'slope'] = slope
    # print(df['slope'])
    # print("len",len(df['slope']))
    return df['slope'].tolist()