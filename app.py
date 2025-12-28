import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import json

# ================= 1. 配置 AIGC Agent (破關版) =================
def generate_food_report(food_name):
    api_key = st.secrets["GEMINI_API_KEY"]
    
    # 嘗試三個最可能的網址路徑
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    ]
    
    payload = {
        "contents": [{"parts": [{"text": f"這是一份「{food_name}」，請用 50 字介紹特色。"}]}]
    }
    
    for url in endpoints:
        try:
            response = requests.post(url, json=payload, timeout=5)
            result = response.json()
            if 'candidates' in result:
                return result['candidates'][0]['content']['parts'][0]['text']
        except:
            continue
            
    return "AI 連線仍然受阻，請確認已使用『New Project』產生的 API Key，並執行 Reboot App。"

# ================= 2. 影像辨識與介面 =================
@st.cache_resource
def load_dl_model():
    return MobileNetV2(weights='imagenet')

dl_model = load_dl_model()

st.title("🍔 食物辨識智能 Agent (終極修復版)")

uploaded_file = st.file_uploader("上傳食物照片", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, use_container_width=True)
    
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = dl_model.predict(x)
    food_name = decode_predictions(preds, top=1)[0][0][1]
    
    st.success(f"辨識結果：{food_name}")
    
    with st.spinner('AI 正在最後嘗試...'):
        st.write(generate_food_report(food_name))