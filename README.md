<!--
 * @Author: hibana2077 hibana2077@gmail.com
 * @Date: 2022-12-23 15:44:56
 * @LastEditors: hibana2077 hibana2077@gmail.com
 * @LastEditTime: 2023-01-02 22:31:56
 * @FilePath: \OOP-independent-study\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->

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

這是使用pytorch實現的基於`LTSM和GRU和MLP`的加密貨幣行情預測系統，使用的數據透過CCXT套件從Binance交易所獲取，並使用pandas套件進行數據處理，使用poltly套件進行數據視覺化，使用Streamlit套件進行網頁化呈現。

## 開發過程及心得

在這個專案中可以大致上分為兩階段，第一階段是數據的爬取和處理以及模型的訓練，第二階段是網頁化呈現並且將模型部署到雲端。

### 第一階段

`Binance API`是加密貨幣交易所`Binance`提供的一種接口，可以通過它來獲取市場數據、交易資訊以及賬戶資訊。`CCXT`是一個`open source`的跨市場加密貨幣交易所套件，支持多種市場和交易協議。通過使用`CCXT`套件，可以用比較簡單的方式訪問`Binance API`，並進行數據搜集以及數據處理。

數據處理的部分使用了`pandas` `sklearn` `talib` 三個套件，其中pandas套件用於數據的讀取和處理，sklearn套件用於數據的標準化，talib套件用於計算技術指標。

模型訓練的部分使用了`pytorch`套件，其中使用了`LSTM`和`GRU`兩種模型，並且使用了`MLP`模型進行模型的融合。

### 第二階段

在第二階段中，我們使用了`Streamlit`套件進行網頁化呈現，使其更加方便使用者使用，並且將模型部署到`GCP`雲端，讓使用者可以透過網路直接使用模型。