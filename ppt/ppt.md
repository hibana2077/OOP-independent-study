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

這是使用pytorch實現的基於`GRU和MLP`的加密貨幣行情預測系統，使用的數據透過CCXT套件從Binance交易所獲取，並使用pandas套件進行數據處理，使用poltly套件進行數據視覺化，使用Streamlit套件進行網頁化呈現。

如果你想要使用這個專案，請先安裝`python3.10`，然後使用`pip`安裝`requirements.txt`中的套件，最後使用`streamlit run app.py`來啟動網頁。或是使用`docker`來啟動網頁，可以去pull這份 [dockerimage](https://hub.docker.com/repository/docker/hibana2077/oop_pj) 然後使用`docker run -p 8501:8501 hibana2077/oop_pj`來啟動網頁。

或是可以直接使用我們的網頁 [link](https://cryptocurrency-predict.herokuapp.com/)

---

# 程式碼講解

### 爬取數據

<!-- 這裡使用 ccxt的binance類別來讀取加密貨幣歷史價格。
並且轉換成pandas dataframe -->

![w:900](https://media.discordapp.net/attachments/868759966431973416/1063626299442212874/image.png)

---

### 數據處理

<!-- 這裡使用talib套件，talib套件提供多種技術指標使用，第一張是建立技術指標，第二張則是把UP DOWN標示出來 -->

![w:900](https://media.discordapp.net/attachments/868759966431973416/1063627280900968478/image.png)

![w:900](https://media.discordapp.net/attachments/868759966431973416/1063627413310943322/image.png)

---

### 正規化

<!-- 把所有要輸入的東西做正規化，這樣的好處是避免梯度爆炸以及可以讓loss小一點。
公式: value - min/max-min-->

![w:1000](https://media.discordapp.net/attachments/868759966431973416/1063628797758419056/image.png?width=1333&height=605)

---

### 分割數據

<!-- 把第1項到12項設定為X，後兩項為Y -->

![](https://media.discordapp.net/attachments/868759966431973416/1063629108258541702/image.png)

---

### 建立資料集類別

<!-- 這裡用sklearn把XY分為測試集 驗證集 訓練集 比例為: 6:3:1 -->

![w:900](https://media.discordapp.net/attachments/868759966431973416/1063629680927846451/image.png)

![w:900](https://media.discordapp.net/attachments/868759966431973416/1063629829066461264/image.png)

---

![](https://media.discordapp.net/attachments/868759966431973416/1063630420698202112/image.png?width=456&height=638)

![bg right:55% 100%](https://media.discordapp.net/attachments/868759966431973416/1063630630056902737/image.png)

---

### 建立模型

<!-- 這個是用來把GRU的第一項作為下一層的輸入，因為GRU的輸出為 y_out , h0(隱藏狀態) -->

#### SelectItem

![w:700](https://media.discordapp.net/attachments/868759966431973416/1063631170492968980/image.png?width=960&height=638)

---

#### Ver1 (MLP+Dropout)

- 基底為MLP
- 使用Dropout避免過擬合

![bg right](https://media.discordapp.net/attachments/868759966431973416/1063631641291014165/image.png?width=517&height=638)


---

#### Ver3 (MLP+GRU)

- 基底為MLP
- 使用Dropout避免過擬合
- 使用GRU來加強過去的資訊

![bg right:60%](https://media.discordapp.net/attachments/868759966431973416/1063632330771677184/image.png?width=832&height=638)

---

#### Ver5 (DUE channel MLP+Dropout)

- 基底為MLP
- 在第二層分成兩個通道
- 一個通道使用較多的Dropout
- 另一個通道使用較少的Dropout
- 並且在最後二層的輸出上使用`torch.cat`來合併

![bg right:45%](https://media.discordapp.net/attachments/868759966431973416/1063633311370575923/image.png?width=436&height=638)

---

<!-- optimizer.zero_grad() 將前一回的損失歸0
     output = model(data) 獲得輸出
     loss = criterion(output, target) 取得輸出跟答案的loss
     loss.backward() 反向傳遞
     optimizer.step() 更新參數
     -->

### 訓練模型(函數)

![w:1150](https://media.discordapp.net/attachments/868759966431973416/1063633949194194985/image.png?width=1302&height=638)

---

<!--
    test_loss += criterion(output, target).item() 將loss加總
    pred = output.data.max(1, keepdim=True)[1] 取得輸出的最大值
    correct += pred.eq(target.data.view_as(pred)).cpu().sum() 取得正確的數量
-->

### 測試模型(函數)

![w:1000](https://media.discordapp.net/attachments/868759966431973416/1063634263347572776/image.png?width=1132&height=637)

---

### 訓練模型(model - ver1)

<!-- 
    batch_size = 64 每次訓練的數量
    shuffle = True 是否打亂資料
    num_workers = 0 使用幾個線程
-->

![w:1100](https://media.discordapp.net/attachments/868759966431973416/1063634510228508752/image.png?width=1178&height=638)

---

### 訓練結果(model - ver1)

#### Loss

![w:650](https://media.discordapp.net/attachments/868759966431973416/1063634870464692354/image.png)

---

#### Accuracy

![w:650](https://media.discordapp.net/attachments/868759966431973416/1063634852441768027/image.png)

---

#### confusion matrix

![bg right:68% w:800](https://media.discordapp.net/attachments/868759966431973416/1063634893818568835/image.png)

---

### 訓練模型(model - ver3)

![w:1100](https://media.discordapp.net/attachments/868759966431973416/1063636587839230001/image.png?width=1178&height=638)

---

### 訓練結果(model - ver3)

#### Loss

![w:650](https://media.discordapp.net/attachments/868759966431973416/1063636752717336596/image.png)

---

#### Accuracy

![w:650](https://media.discordapp.net/attachments/868759966431973416/1063636763068858388/image.png)

---

#### confusion matrix

![bg right:68% w:800](https://media.discordapp.net/attachments/868759966431973416/1063636782211665920/image.png)

---

### 訓練模型(model - ver5)

![w:1100](https://media.discordapp.net/attachments/868759966431973416/1063637343531171890/image.png?width=1178&height=638)

---

### 訓練結果(model - ver5)

#### Loss

![w:650](https://media.discordapp.net/attachments/868759966431973416/1063637453983985744/image.png)

---

#### Accuracy

![w:650](https://media.discordapp.net/attachments/868759966431973416/1063637468454322186/image.png)

---

#### confusion matrix

![bg right:68% w:800](https://media.discordapp.net/attachments/868759966431973416/1063637494115090503/image.png)

---

# 結論

透過以上實驗，我們可以發現，要預測加密貨幣價格走勢是一件不太容易的事情，因為加密貨幣的價格走勢是一個非常不穩定的東西，而且價格走勢的變化也是非常快速的，因此我們在訓練模型的時候，需要將資料集的時間間隔設定得越短越好，這樣才能讓模型更好的去預測價格走勢，我們也發現，模型的深度太大的話資料必須要增加，才能提升模型的準確度。

---

# 部屬模型

模型訓練完成後，我們就可以將模型部屬到網站上，網站透過streamlit框架完成，並且使用firebase平台部屬，網站的功能是可以輸入加密貨幣的名稱，然後輸入想要預測的時間，就可以預測該加密貨幣的價格走勢。

---

# 網站介紹


---

# 使用方式

![bg right:68% w:800](https://media.discordapp.net/attachments/868759966431973416/1063660519405391942/image.png?width=1333&height=559)

- 選擇 `模型使用` 頁面
- 選擇想要預測的加密貨幣
- 輸入想要預測的時間
- 點擊 `預測` 按鈕

---

# 預測結果

![bg right:68% w:750](https://media.discordapp.net/attachments/868759966431973416/1063661538189254798/image.png?width=689&height=638)

---

# 參考資料

[1] [https://www.kaggle.com/](https://www.kaggle.com/)

[2] [PyTorch 中文手册(pytorch handbook)](https://github.com/zergtant/pytorch-handbook)

[3] [PyTorch 官方文件](https://pytorch.org/docs/stable/index.html)

[4] [LSTM-Classification-pytorch](https://github.com/jiangqy/LSTM-Classification-pytorch)

[5] [CCXT 官方文件](https://docs.ccxt.com/en/latest/manual.html)

--- 

<!-- _class: lead -->

# 謝謝大家

powered by [marp](https://marp.app)