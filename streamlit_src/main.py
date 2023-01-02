'''
Author: hibana2077 hibana2077@gmail.com
Date: 2022-12-23 15:45:40
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-01-02 21:56:47
FilePath: \OOP-independent-study\streamlit_src\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import streamlit as st
import pickle
import ccxt
import pandas as pd
import os
import torch
import torch.nn as nn
import plotly.graph_objects as go
from talib import abstract
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu


st.set_page_config(layout="wide")

class SelectItem(torch.nn.Module):#這是用來取出多個輸出其中一個的輸出，如果不用sequential的話，就可以不用這個
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class crypto_classfier_ver5(nn.Module):#CNN+GRU+MLP
    def __init__(self):
        super(crypto_classfier_ver5, self).__init__()
        self.name = "CCV-5"
        self.net = nn.Sequential(
            torch.nn.Conv1d(10, 20, 3, stride=1, padding=1),
            torch.nn.ELU(),
            torch.nn.MaxPool1d(1, stride=1),
            torch.nn.Conv1d(20, 40, 3, stride=1, padding=1),
            torch.nn.ELU(),
            torch.nn.MaxPool1d(1, stride=1),
            torch.nn.Conv1d(40, 1, 3, stride=1, padding=1),
            torch.nn.Linear(12,64),
            torch.nn.Linear(64,128),
            torch.nn.GRU(128, 64, 25),
            SelectItem(0),
            torch.nn.Dropout(0.5),
            torch.nn.GRU(64, 32, 25),
            SelectItem(0),
            torch.nn.Dropout(0.5),
            torch.nn.GRU(32, 16, 25),
            SelectItem(0),
            SelectItem(0),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(16,2),
            torch.nn.Softmax(dim=0)
            )
    def forward(self, x):
        x = self.net(x)
        return x

def home():
    st.write("""
    
# 物件導向期末專題 - 基於LSTM的MLP加密貨幣買賣點預測系統

![GitHub](https://img.shields.io/github/license/hibana2077/OOP-independent-study?style=plastic-square)
![GitHub repo size](https://img.shields.io/github/repo-size/hibana2077/OOP-independent-study?style=plastic-square)
![GitHub language count](https://img.shields.io/github/languages/count/hibana2077/OOP-independent-study?style=plastic-square)
![GitHub top language](https://img.shields.io/github/languages/top/hibana2077/OOP-independent-study?style=plastic-square)

## 介紹

![python](https://img.shields.io/badge/python-3.10-blue?style=plastic-square&logo=python)
![pytorch](https://img.shields.io/badge/pytorch-1.13.1-EE4C2C?style=plastic-square&logo=pytorch)
![pandas](https://img.shields.io/badge/pandas-1.3.4-150458?style=plastic-square&logo=pandas)
![Poltly](https://img.shields.io/badge/poltly-5.3.1-3F4F75?style=plastic-square&logo=Plotly)
![Streamlit](https://img.shields.io/badge/streamlit-1.2.0-FF4B4B?style=plastic-square&logo=streamlit)
![Binance](https://img.shields.io/badge/binance-API-2F3336?style=plastic-square&logo=binance)

這是使用pytorch實現的基於`LTSM和GRU和MLP`的加密貨幣行情預測系統，使用的數據透過CCXT套件從Binance交易所獲取，並使用pandas套件進行數據處理，使用poltly套件進行數據視覺化，使用Tensorboard套件進行模型訓練過程視覺化，使用Streamlit套件進行網頁化呈現。

## 開發過程及心得


    """)

def data_4_RFCseries(df:pd.DataFrame):
    df['MFI'] = abstract.MFI(df, timeperiod=14)
    df['RSI'] = abstract.RSI(df, timeperiod=14)
    df['ADX'] = abstract.ADX(df, timeperiod=14)
    df['CCI'] = abstract.CCI(df, timeperiod=14)
    df['ATR'] = abstract.ATR(df, timeperiod=14)
    df['OBV'] = abstract.OBV(df, timeperiod=14)
    df['EMA'] = abstract.EMA(df, timeperiod=14)
    df['WILLR'] = abstract.WILLR(df, timeperiod=14)
    df['AD'] = abstract.AD(df, timeperiod=14)
    df['ADOSC'] = abstract.ADOSC(df, timeperiod=14)
    df['ADXR'] = abstract.ADXR(df, timeperiod=14)
    df['APO'] = abstract.APO(df, timeperiod=14)
    df['AROONOSC'] = abstract.AROONOSC(df, timeperiod=14)
    df['BOP'] = abstract.BOP(df, timeperiod=14)
    df['CCI'] = abstract.CCI(df, timeperiod=14)
    df['CMO'] = abstract.CMO(df, timeperiod=14)
    df['DX'] = abstract.DX(df, timeperiod=14)
    df['MOM'] = abstract.MOM(df, timeperiod=14)
    df['PPO'] = abstract.PPO(df, timeperiod=14)
    df['ROC'] = abstract.ROC(df, timeperiod=14)
    df['ROCP'] = abstract.ROCP(df, timeperiod=14)
    df['ROCR'] = abstract.ROCR(df, timeperiod=14)
    df['ROCR100'] = abstract.ROCR100(df, timeperiod=14)
    df['RSI'] = abstract.RSI(df, timeperiod=14)
    df['TRIX'] = abstract.TRIX(df, timeperiod=14)
    df['ULTOSC'] = abstract.ULTOSC(df, timeperiod=14)
    df['WILLR'] = abstract.WILLR(df, timeperiod=14)
    df['WMA'] = abstract.WMA(df, timeperiod=14)
    df['HT_TRENDLINE'] = abstract.HT_TRENDLINE(df, timeperiod=14)
    df['TRANGE'] = abstract.TRANGE(df, timeperiod=14)

    minmaxsc = MinMaxScaler()
    df = df.dropna()
    df = df.drop(['time'], axis=1)
    df = minmaxsc.fit_transform(df)
    #取最後一筆作為提問
    x = df[-1]
    x = x.reshape(1, -1)
    #讀取模型
    model = pickle.load(open('model/RFCV1.sav', 'rb'))
    #預測
    y = model.predict(x)
    st.write("預測結果")
    st.write(y[0])
    output = "不會上漲📉" if y == 0 else "會上漲📈"
    if y[0] == 0:
        st.markdown("<h1 style='text-align: center; color: red;'>{}</h1>".format(output), unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: green;'>{}</h1>".format(output), unsafe_allow_html=True)

def data_4_CCSeries(df:pd.DataFrame,model_name:str):
    df['RSI'] = abstract.RSI(df, timeperiod=14)
    df['MACD'] = abstract.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)['macd'] #只取MACD
    df['OBV'] = abstract.OBV(df, timeperiod=14)
    df['CCI'] = abstract.CCI(df, timeperiod=14)
    df['ATR'] = abstract.ATR(df, timeperiod=14)
    df['ADX'] = abstract.ADX(df, timeperiod=14)
    df['MFI'] = abstract.MFI(df, timeperiod=14)

    minmaxsc = MinMaxScaler()
    df = df.dropna()
    df = df.drop(['time'], axis=1)
    df = minmaxsc.fit_transform(df)

    #取得提問資料
    X = df[-10:, 0:13]

    #讀取模型
    model = torch.load(f'model/{model_name}',map_location=torch.device('cpu'))
    #預測
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    y = model(X)
    up,dn = y[0],y[1]
    output = "不會上漲📉" if up < dn else "會上漲📈"
    if y[0] == 0:
        st.markdown("<h1 style='text-align: center; color: red;'>{}</h1>".format(output), unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: green;'>{}</h1>".format(output), unsafe_allow_html=True)

def data_process(model_name:str , exchange:ccxt.Exchange, symbol , timeframe):
    st.write("下載數據中...")
    data = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe)
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    #show candlestick
    st.write("K線圖")
    candlestick = go.Figure(data=[go.Candlestick(x=df['time'],
                                                 open=df['open'],
                                                 high=df['high'],
                                                 low=df['low'],
                                                 close=df['close'])])
    st.plotly_chart(candlestick)
    if model_name.startswith('RFCV'):
        data_4_RFCseries(df)
    if model_name.startswith('CCV'):
        data_4_CCSeries(df,model_name)
        
#HI this is a test

def model():
    map_table = {
        'Binance': ccxt.binance({
            'options': {
                'defaultType': 'spot',
            },
        }),
        'Huobi': ccxt.huobi({
            'options': {
                'defaultType': 'spot',
            },
        }),
        'Kraken': ccxt.kraken({
            'options': {
                'defaultType': 'spot',
            },
        }),
        'OKEx': ccxt.okex({
            'options': {
                'defaultType': 'spot',
            },
        }),
        'Cryptocom' : ccxt.cryptocom({
            'options': {
                'defaultType': 'spot',
            },
        }),
    }
    st.title('模型使用📁')
    modes = os.listdir('model')
    mode = st.selectbox('選擇模型', modes)
    exchange_input = st.selectbox('選擇交易所', ['Binance', 'Huobi', 'Kraken', 'OKEx','Bitget','Cryptocom'])
    exchange:ccxt.Exchange = map_table[exchange_input]
    exchange.load_markets()
    symbol_input = st.selectbox('選擇貨幣', list(exchange.symbols))
    symbol = symbol_input
    timeframe_input = st.selectbox('選擇時間間隔', ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M'])
    timeframe = timeframe_input
    st.write('選擇的交易所:', exchange_input)
    st.write('選擇的貨幣:', symbol)
    st.write('選擇的時間間隔:', timeframe)
    if st.button('開始預測'):
        st.write('開始預測...')
        data_process(mode, exchange, symbol, timeframe)

def technical():
    st.title('技術介紹')
    st.write()

def about():
    st.title('成員')
    st.markdown('## 資工系 資工二乙')
    st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">李軒豪</p>', unsafe_allow_html=True)
    st.write('- [Github](https://github.com/hibana2077)','🐙')
    st.write('- [Portfolio](https://hibana2077-f1fa1.web.app/)', '📄')
    st.markdown('<p class="big-font">丁敬原</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">林品豪</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">甘佳峻</p>', unsafe_allow_html=True)
    

#建立分頁
PAGES = {
    "首頁": home,
    "模型使用": model,
    "技術介紹": technical,
    "成員": about
}

with st.sidebar:
    selection = option_menu("走勢預測機器人", list(PAGES.keys()), 
        icons=['house', 'list-task', 'braces', 'laptop'],menu_icon='robot',default_index=1,orientation='vertical'
    )
page = PAGES[selection]
page()
