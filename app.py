import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import streamlit as st


import requests
import json

# ================= 1. 配置 AIGC Agent (直接使用 API) =================
def generate_food_report(food_name):
    api_key = st.secrets["GEMINI_API_KEY"]
    # 直接使用正式版 v1 API 路徑，避開 SDK 的 v1beta 錯誤
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"你是一個專業的美食評論家。影像辨識模型判斷這是一份「{food_name}」。請用 100 字以內介紹它的特色，並列出主要營養成分。"
            }]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()
        # 取得回傳的文字內容
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"AI 報告生成失敗：{str(e)}"



# ================= 2. 載入深度學習模型 (方法一) =================
@st.cache_resource
def load_dl_model():
    # 使用 MobileNetV2，預訓練權重為 imagenet
    return MobileNetV2(weights='imagenet')

model = load_dl_model()

# ================= 3. Streamlit 介面設計 =================
st.title("🍔 Taica AIGC 課程專題：食物辨識智能 Agent")
st.write("上傳一張食物照片，深度學習模型將進行辨識，並由 AI Agent 撰寫評論。")

uploaded_file = st.file_uploader("選擇一張圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 顯示圖片
    img = Image.open(uploaded_file)
    st.image(img, caption='上傳的圖片', use_container_width=True)
    
    # 影像預處理
    # 1. 強制轉為 RGB (避免 PNG 的 4 通道問題)
    img_rgb = img.convert('RGB')
    
    # 2. 調整大小為 MobileNetV2 要求的 224x224
    img_resized = img_rgb.resize((224, 224))
    
    # 3. 轉為 Numpy 陣列
    x = image.img_to_array(img_resized)
    
    # 4. 增加批次維度，從 (224, 224, 3) 變成 (1, 224, 224, 3)
    x = np.expand_dims(x, axis=0)
    
    # 5. 執行 MobileNetV2 的專屬預處理 (包含數值縮放)
    x = preprocess_input(x)

    # 執行辨識
    with st.spinner('深度學習模型辨識中...'):
        preds = model.predict(x)
        # 取得最高機率的結果 (Label)
        results = decode_predictions(preds, top=1)[0]
        food_name_en = results[0][1] # 取得英文名稱
        confidence = results[0][2]

    st.success(f"辨識結果：{food_name_en} (信心度: {confidence:.2%})")

    # 執行 Agent 延伸功能
    st.divider()
    st.subheader("🤖 AI Agent 延伸報告")
    with st.spinner('Agent 正在撰寫文案...'):
        # 這裡可以加入一個簡單的翻譯或直接把英文名給 LLM
        report = generate_food_report(food_name_en)
        st.write(report)

st.divider()
st.caption("參考來源：Taica AIGC 課程實作 | 模型：MobileNetV2 | 部署：Streamlit")