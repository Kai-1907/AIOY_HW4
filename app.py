import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import json

# ================= 1. 配置 AIGC Agent (避開 404 路由版) =================
def generate_food_report(food_name):
    api_key = st.secrets["GEMINI_API_KEY"]
    
    # 這是最後一招：直接使用最原始的「文本生成」專用節點，不帶任何 beta 或 v1 分支測試
    # 如果這個節點再說找不到模型，代表該 API Key 需要在 AI Studio 重新建立一個「全新的 Project」
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": f"你是一個專業的美食評論家。這是一份「{food_name}」。請用 100 字介紹特色與營養。"}]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        result = response.json()
        
        if 'candidates' in result:
            return result['candidates'][0]['content']['parts'][0]['text']
        
        # 如果還是失敗，嘗試強制切換到 gemini-1.0-pro (這是最老、最不可能找不到的模型)
        url_backup = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
        response = requests.post(url_backup, headers=headers, data=json.dumps(payload), timeout=10)
        result = response.json()
        
        if 'candidates' in result:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"連 Google 伺服器都找不到模型，建議重新至 AI Studio 產生新 Key。錯誤：{result.get('error', {}).get('message')}"
    except Exception as e:
        return f"連線異常：{str(e)}"

# ================= 2. 載入模型與其餘介面 (保持不變) =================
@st.cache_resource
def load_dl_model():
    return MobileNetV2(weights='imagenet')

model = load_dl_model()

st.title("🍔 食物辨識智能 Agent")
uploaded_file = st.file_uploader("選擇圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, use_container_width=True)
    img_resized = img.convert('RGB').resize((224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img_resized), axis=0))
    
    preds = model.predict(x)
    food_name = decode_predictions(preds, top=1)[0][0][1]
    st.success(f"辨識結果：{food_name}")
    
    with st.spinner('AI 正在嘗試最後的連線...'):
        st.write(generate_food_report(food_name))