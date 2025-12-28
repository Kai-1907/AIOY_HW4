import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import json

# ================= 1. AI 報告生成 (三路徑輪詢強攻版) =================
def generate_food_report(food_name):
    api_key = st.secrets["GEMINI_API_KEY"]
    
    # 同時嘗試三種可能的 API 路徑，只要一條通了就行
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    ]
    
    payload = {"contents": [{"parts": [{"text": f"你是一個專業美食評論家。辨識結果是「{food_name}」。請寫 50 字介紹特色。"}]}]}
    
    for url in endpoints:
        try:
            response = requests.post(url, json=payload, timeout=8)
            result = response.json()
            if 'candidates' in result:
                return result['candidates'][0]['content']['parts'][0]['text']
        except:
            continue
    return "AI 報告生成失敗：即便換了新 Key，所有路徑仍回傳 404。請確認 API Key 狀態。"

# ================= 2. 影像辨識 (保持原有的成功邏輯) =================
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

st.title("🍔 食物辨識智能 Agent (終極修復)")
uploaded_file = st.file_uploader("選擇照片...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, use_container_width=True)
    
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = model.predict(x)
    food_name = decode_predictions(preds, top=1)[0][0][1]
    
    st.success(f"辨識結果：{food_name}")
    
    with st.spinner('AI 正在嘗試最後的連線...'):
        st.write(generate_food_report(food_name))