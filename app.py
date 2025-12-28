import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from google import genai
import streamlit as st

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ================= 1. 配置 AIGC Agent (純 Web API 版) =================
def generate_food_report(food_name):
    prompt = f"""
你是一個專業的美食評論家。
影像辨識模型判斷這是一份「{food_name}」。
請用 100 字以內介紹它的特色，並列出主要營養成分。
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )

    return response.text


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