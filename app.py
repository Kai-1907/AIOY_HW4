import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import json

# ================= 1. 配置 AI Agent (完全繞過 SDK) =================
def generate_food_report(food_name):
    api_key = st.secrets["GEMINI_API_KEY"]
    # 改用全球通用的正式版 v1beta 網址，這在 Streamlit Cloud 最穩定
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": f"你是一個美食家。辨識結果是「{food_name}」。請寫 50 字介紹與營養。"}]}]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        result = response.json()
        if 'candidates' in result:
            return result['candidates'][0]['content']['parts'][0]['text']
        return f"AI 服務暫忙 (錯誤碼: {result.get('error', {}).get('code')})"
    except:
        return "連線逾時，請重試。"

# ================= 2. 影像辨識 (保持您原本成功的邏輯) =================
@st.cache_resource
def load_dl_model():
    return MobileNetV2(weights='imagenet')

model = load_dl_model()

st.title("🍔 食物辨識智能 Agent")
uploaded_file = st.file_uploader("選擇照片...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, use_container_width=True)
    
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = model.predict(x)
    food_name = decode_predictions(preds, top=1)[0][0][1]
    
    st.success(f"辨識結果：{food_name}")
    st.write(generate_food_report(food_name))