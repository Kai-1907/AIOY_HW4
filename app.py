import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import json

# ================= 1. 配置 AIGC Agent (避開 SDK 衝突版) =================
def generate_food_report(food_name):
    api_key = st.secrets["GEMINI_API_KEY"]
    
    # 這是全球通用的正式路徑，不依賴任何 SDK 套件
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": f"你是一個專業的美食評論家。辨識結果是「{food_name}」。請寫一段 100 字以內的特色介紹與營養成分。"}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        result = response.json()
        if 'candidates' in result:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"AI 暫時無法回應，辨識結果為：{food_name}"
    except:
        return f"連線異常，辨識結果為：{food_name}"

# ================= 2. 深度學習與介面 (標準流程) =================
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

st.title("🍔 食物辨識智能 Agent")
uploaded_file = st.file_uploader("選擇照片...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, use_container_width=True)
    
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = model.predict(x)
    food_name = decode_predictions(preds, top=1)[0][0][1]
    
    st.success(f"辨識結果：{food_name}")
    
    with st.spinner('AI 正在撰寫報告...'):
        st.write(generate_food_report(food_name))