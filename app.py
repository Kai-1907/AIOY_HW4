import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from openai import OpenAI  # 改用 OpenAI 庫

# ================= 1. 配置 OpenAI Agent =================
def generate_food_report(food_name):
    # 從 Secrets 讀取 OpenAI Key
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 這是目前最快且最便宜的模型
            messages=[
                {"role": "system", "content": "你是一個美食評論家。"},
                {"role": "user", "content": f"辨識結果是「{food_name}」。請寫 100 字介紹特色與營養成分。"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI 連線失敗：{str(e)}"

# ================= 2. 影像辨識 (保持不變) =================
@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

model = load_model()

# ================= 3. 介面設計 =================
st.title("🍔 食物辨識智能 Agent (OpenAI 版)")

uploaded_file = st.file_uploader("選擇照片...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, use_container_width=True)
    
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = model.predict(x)
    food_name = decode_predictions(preds, top=1)[0][0][1]
    
    st.success(f"辨識結果：{food_name}")
    
    with st.spinner('OpenAI 正在撰寫報告...'):
        st.write(generate_food_report(food_name))