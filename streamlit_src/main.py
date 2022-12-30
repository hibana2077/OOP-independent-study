'''
Author: hibana2077 hibana2077@gmail.com
Date: 2022-12-23 15:45:40
LastEditors: hibana2077 hibana2077@gmail.com
LastEditTime: 2022-12-30 13:32:11
FilePath: \OOP-independent-study\streamlit_src\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import streamlit as st

model_file_url = '/model/type1.pkl'

def home():
    st.title('首頁')
    st.write('This is the **Home** page')

def model():
    st.title('模型')
    st.write('This is the **Model** page')

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
    st.write('- [Github](https://github.com/hibana2077) :octocat:')
    st.write('- [Portfolio](https://hibana2077-f1fa1.web.app/)')
    st.markdown('<p class="big-font">丁敬原</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">林品豪</p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font">甘佳峻</p>', unsafe_allow_html=True)
    

#建立分頁
PAGES = {
    "首頁": home,
    "模型": model,
    "成員": about
}

st.sidebar.title('分頁')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))   
page = PAGES[selection]
page()