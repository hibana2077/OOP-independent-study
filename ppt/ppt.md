---
marp: true
theme: gaia
class: invert
---

<!-- _backgroundImage: url('https://marp.app/assets/hero-background.jpg')-->
<!-- _class: lead -->

# 物件導向期末專題報告

第8組

---

# 主題 : 加密貨幣行情預測系統

我們之所以會選擇這個主題，是因為我們覺得這個主題可以讓我們學習到很多新的東西，並且可以讓我們在未來的工作上有所幫助。

---

# 介紹

這是使用pytorch實現的基於`LTSM和GRU和MLP`的加密貨幣行情預測系統，使用的數據透過CCXT套件從Binance交易所獲取，並使用pandas套件進行數據處理，使用poltly套件進行數據視覺化，使用Streamlit套件進行網頁化呈現。

如果你想要使用這個專案，請先安裝`python3.10`，然後使用`pip`安裝`requirements.txt`中的套件，最後使用`streamlit run app.py`來啟動網頁。或是使用`docker`來啟動網頁，可以去pull這份 [dockerimage](https://hub.docker.com/repository/docker/hibana2077/oop_pj) 然後使用`docker run -p 8501:8501 hibana2077/oop_pj`來啟動網頁。

或是可以直接使用我們的網頁 [link](https://cryptocurrency-predict.herokuapp.com/)

---

# 程式碼講解

### 爬取數據

![w:900](https://media.discordapp.net/attachments/868759966431973416/1063626299442212874/image.png)

---

### 數據處理

<!-- This is a presenter note for this page. -->

![w:900](https://media.discordapp.net/attachments/868759966431973416/1063627280900968478/image.png)

![w:900](https://media.discordapp.net/attachments/868759966431973416/1063627413310943322/image.png)

---

