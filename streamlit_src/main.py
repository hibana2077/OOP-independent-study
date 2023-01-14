'''
Author: hibana2077 hibana2077@gmail.com
Date: 2022-12-23 15:45:40
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2023-01-14 11:25:31
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
import torch.nn.functional as F
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

class CCV5(nn.Module):#DUE channel MLP
    def __init__(self):
        super(CCV5, self).__init__()
        self.pub1 = nn.Linear(12,32)
        self.a1layer1 = nn.Linear(32, 256)
        self.a1layer2 = nn.Linear(256, 512)
        self.a1layer3 = nn.Linear(512, 256)
        self.a1layer4 = nn.Linear(256, 64)#a1 end
        self.a2layer1 = nn.Linear(32, 256)
        self.a2layer2 = nn.Linear(256, 556)
        self.a2layer3 = nn.Linear(556, 256)
        self.a2layer4 = nn.Linear(256, 192)#a2 end
        self.concat = nn.Linear(256, 32)
        self.a3layer1 = nn.Linear(32, 16)
        self.a3layer2 = nn.Linear(16, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.pub1(x))
        a1 = F.relu(self.a1layer1(x))
        a1 = F.relu(self.a1layer2(a1))
        a1 = self.dropout(a1)
        a1 = F.relu(self.a1layer3(a1))
        a1 = F.relu(self.a1layer4(a1))
        a2 = F.relu(self.a2layer1(x))
        a2 = F.relu(self.a2layer2(a2))
        a2 = self.dropout(a2)
        a2 = F.relu(self.a2layer3(a2))
        a2 = F.relu(self.a2layer4(a2))
        x = torch.cat((a1, a2), 1)
        x = F.relu(self.concat(x))
        x = F.relu(self.a3layer1(x))
        x = self.a3layer2(x)
        return x

class CCV3(nn.Module):#MLP+GRU
    def __init__(self):
        super(CCV3, self).__init__()
        self.Linear1 = nn.Linear(12, 128)
        self.Linear2 = nn.Linear(128, 256)
        self.Linear3 = nn.Linear(256, 2)
        self.Dropout1 = nn.Dropout(0.168)
        self.GRU1 = nn.GRU(256, 256, 1, batch_first=True)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Dropout1(x)
        x , _ = self.GRU1(x)
        x = self.Linear3(x)
        return x

class CCV1(nn.Module):#MLP
    def __init__(self):
        super(CCV1, self).__init__()
        self.Linear1 = nn.Linear(12, 128)
        self.Linear2 = nn.Linear(128, 256)
        self.Linear3 = nn.Linear(256, 1024)
        self.Linear4 = nn.Linear(1024, 256)
        self.Linear5 = nn.Linear(256, 128)
        self.Linear6 = nn.Linear(128, 64)
        self.Linear7 = nn.Linear(64, 2)
        self.Dropout1 = nn.Dropout(0.3)
            
    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        x = self.Dropout1(x)
        x = F.relu(self.Linear3(x))
        x = F.relu(self.Linear4(x))
        x = self.Dropout1(x)
        x = F.relu(self.Linear5(x))
        x = F.relu(self.Linear6(x))
        x = self.Dropout1(x)
        x = self.Linear7(x)
        return x

def home():
    st.write("""

# 物件導向期末專題 - 基於LSTM的MLP加密貨幣行情預測系統

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

這是使用pytorch實現的基於`LTSM和GRU和MLP`的加密貨幣行情預測系統，使用的數據透過CCXT套件從Binance交易所獲取，並使用pandas套件進行數據處理，使用poltly套件進行數據視覺化，使用Streamlit套件進行網頁化呈現。

## 開發過程及心得

在這個專案中可以大致上分為兩階段，第一階段是數據的爬取和處理以及模型的訓練，第二階段是網頁化呈現並且將模型部署到雲端。

### 第一階段

`Binance API`是加密貨幣交易所`Binance`提供的一種接口，可以通過它來獲取市場數據、交易資訊以及賬戶資訊。`CCXT`是一個`open source`的跨市場加密貨幣交易所套件，支持多種市場和交易協議。通過使用`CCXT`套件，可以用比較簡單的方式訪問`Binance API`，並進行數據搜集以及數據處理。

數據處理的部分使用了`pandas` `sklearn` `talib` 三個套件，其中pandas套件用於數據的讀取和處理，sklearn套件用於數據的標準化，talib套件用於計算技術指標。

模型訓練的部分使用了`pytorch`套件，其中使用了`LSTM`和`GRU`兩種模型，並且使用了`MLP`模型進行模型的融合。

### 第二階段

在第二階段中，我們使用了`Streamlit`套件進行網頁化呈現，使其更加方便使用者使用，並且將模型部署到`GCP`雲端，讓使用者可以透過網路直接使用模型。

### 心得

在這個專案中，我們這一組使用了這一年最常聽到的東西作為主題，在製作過程中也有遇到一些問題，像是`pytorch`的結構跟`Tensorflow`不太一樣，導致我們在訓練模型的時候遇到了一些問題，像是模型建立方式上的差異跟訓練方式上的差異，也遇到了梯度爆炸跟過擬合的問題，但是靠著`Github`跟`HackMD`上的資訊以及`pytorch`的官方文件，我們最後還是解決了這些問題，並且將模型部署到雲端，讓使用者可以透過網路直接使用模型。


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
    #釋放model占用的記憶體
    del model

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
    path = os.path.join(os.getcwd(), 'model', model_name)
    model_CCV = torch.load(path)
    #預測
    model_CCV.eval()
    X = torch.tensor(X, dtype=torch.float32)
    y = model_CCV(X)[-1]
    st.write("預測結果")
    out = "不會上漲📉" if y.argmax().item() else "會上漲📈"
    if y[0] == 0:
        st.markdown("<h1 style='text-align: center; color: red;'>{}</h1>".format(out), unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: green;'>{}</h1>".format(out), unsafe_allow_html=True)
    #釋放model占用的記憶體
    del model_CCV

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

def model_cooose():
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
    st.write("""
# 技術使用

## 1. 爬蟲

![Binance](https://img.shields.io/badge/binance-API-2F3336?style=plastic-square&logo=binance)

## 2. 資料處理

![Python](https://img.shields.io/badge/python-3.10-2F3336?style=plastic-square&logo=python)

![Pandas](https://img.shields.io/badge/pandas-1.3.4-150458?style=plastic-square&logo=pandas)

![Numpy](https://img.shields.io/badge/numpy-1.21.2-013243?style=plastic-square&logo=numpy)

## 3. 視覺化

![Streamlit](https://img.shields.io/badge/streamlit-1.2.0-FF4B4B?style=plastic-square&logo=streamlit)

![Poltly](https://img.shields.io/badge/poltly-5.3.1-3F4F75?style=plastic-square&logo=Plotly)

## 4. 部署

![GCP](https://img.shields.io/badge/GCP-Cloud-4285F4?style=plastic-square&logo=google-cloud)

![Docker](https://img.shields.io/badge/docker-20.10.8-2496ED?style=plastic-square&logo=docker)

![Firebase](https://img.shields.io/badge/firebase-front_end-FFCA28?style=plastic-square&logo=firebase)""")

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
    "模型使用": model_cooose,
    "技術介紹": technical,
    "成員": about
}

with st.sidebar:
    selection = option_menu("走勢預測機器人", list(PAGES.keys()), 
        icons=['house', 'list-task', 'braces', 'laptop'],menu_icon='robot',default_index=1,orientation='vertical'
    )
page = PAGES[selection]
page()
