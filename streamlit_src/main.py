'''
Author: hibana2077 hibana2077@gmail.com
Date: 2022-12-23 15:45:40
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2022-12-30 14:03:59
FilePath: \OOP-independent-study\streamlit_src\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import streamlit as st
import ccxt
import pandas as pd


model_file_url = '/model/type1.pkl'
st.set_page_config(layout="wide")

def home():
    st.write("""
    
# 物件導向期末專題 - 基於LSTM的MLP加密貨幣價格預測系統

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

這是使用pytorch實現的加密貨幣價格預測系統，使用的數據透過CCXT套件從Binance交易所獲取，並使用pandas套件進行數據處理，使用poltly套件進行數據視覺化，使用Tensorboard套件進行模型訓練過程視覺化，使用Streamlit套件進行網頁化呈現。


## 開發過程及心得


    """)

def model():
    map_table = {
        'Binance': ccxt.binance(),
        'Bitfinex': ccxt.bitfinex(),
        'Bitstamp': ccxt.bitstamp(),
        'Bittrex': ccxt.bittrex(),
        'Coinbase': ccxt.coinbase(),
        'Coinbase Pro': ccxt.coinbasepro(),
        'Huobi': ccxt.huobi(),
        'Kraken': ccxt.kraken(),
        'OKEx': ccxt.okex(),
        'Poloniex': ccxt.poloniex()
    }
    st.title('模型使用📁')
    exchange_input = st.selectbox('選擇交易所', ['Binance', 'Bitfinex', 'Bitstamp', 'Bittrex', 'Coinbase', 'Coinbase Pro', 'Huobi', 'Kraken', 'OKEx', 'Poloniex'])
    exchange:ccxt.Exchange = map_table[exchange_input]
    exchange.load_markets()
    symbol_input = st.selectbox('選擇貨幣', list(exchange.symbols))
    symbol = symbol_input
    timeframe_input = st.selectbox('選擇時間間隔', ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w', '1M'])
    timeframe = timeframe_input
    limit_input = st.number_input('選擇數據量', min_value=100, max_value=1000, value=100)
    limit = limit_input
    st.write('選擇的交易所:', exchange_input)
    st.write('選擇的貨幣:', symbol_input)
    st.write('選擇的時間間隔:', timeframe_input)
    st.write('選擇的數據量:', limit_input)
    if st.button('開始預測'):
        st.write('開始預測...')

def technical():
    st.title('技術介紹')

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

st.sidebar.title('預測機器人🤖')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))   
page = PAGES[selection]
page()
