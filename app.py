import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import json

# ================= 1. 配置 AIGC Agent (純 Web API 版) =================
def generate_food_report(food_name):
    api_key = st.secrets["GEMINI_API_KEY"]
    
    # 這是目前全球通用的標準正式版網址，避開所有 beta 測試通道的穩定性問題
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"你是一個專業的美食評論家。影像辨識結果是「{food_name}」。請寫一段 100 字以內的美味介紹，並列出主要營養成分。"
            }]
        }]
    }
    
    try:
        # 強制使用 json 格式發送
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()
        
        if 'candidates' in result:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            # 這是 debug 的關鍵：如果還是失敗，請截圖給我看這裡印出的具體錯誤
            error_msg = result.get('error', {}).get('message', '未知錯誤')
            return f"AI 報告失敗：{error_msg} (Debug: {json.dumps(result)})"
    except Exception as e:
        return f"連線異常：{str(e)}"

# ================= 2. 載入深度學習模型 =================
@st.cache_resource
def load_dl_model():
    return MobileNetV2(weights='imagenet')

model = load_dl_model()

# ================= 3. Streamlit 介面設計 =================
st.title("🍔 食物辨識智能 Agent")
st.write("上傳一張食物照片，由 AI Agent 撰寫評論。")

uploaded_file = st.file_uploader("選擇一張圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='上傳的圖片', use_container_width=True)
    
    # 影像預處理
    img_rgb = img.convert('RGB')
    img_resized = img_rgb.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # 進行辨識
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=1)[0]
    food_name_en = decoded_preds[0][1]
    confidence = decoded_preds[0][2]
    
    st.success(f"辨識結果：{food_name_en} (信心度: {confidence:.2%})")
    
    # 呼叫 Gemini Agent
    with st.spinner('AI Agent 正在撰寫報告...'):
        report = generate_food_report(food_name_en)
        st.subheader("🤖 AI Agent 延伸報告")
        st.write(report)