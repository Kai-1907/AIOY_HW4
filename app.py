import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import google.generativeai as genai  # 改用最穩定的 SDK 結構

# ================= 1. 配置 Gemini AI =================
# 從 Secrets 讀取 Key 並進行初始化設定
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def generate_food_report(food_name):
    try:
        # 使用 GenerativeModel 結構，這是目前最不容易報 404 的寫法
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"你是一個專業的美食評論家。影像辨識模型判斷這是一份「{food_name}」。請用 100 字以內介紹它的特色，並列出主要營養成分。"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # 如果 1.5-flash 還是不行，自動降級嘗試 gemini-pro (確保一定有回應)
        try:
            model_backup = genai.GenerativeModel('gemini-pro')
            response = model_backup.generate_content(prompt)
            return response.text
        except Exception as e2:
            return f"AI 報告生成失敗。錯誤訊息：{str(e2)}"

# ================= 2. 載入深度學習模型 (MobileNetV2) =================
@st.cache_resource
def load_dl_model():
    return MobileNetV2(weights='imagenet')

dl_model = load_dl_model()

# ================= 3. Streamlit 介面設計 =================
st.title("🍔 食物辨識智能 Agent (穩定修復版)")
st.write("上傳照片進行辨識，並由 AI 撰寫延伸報告。")

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
    preds = dl_model.predict(x)
    decoded_preds = decode_predictions(preds, top=1)[0]
    food_name_en = decoded_preds[0][1]
    confidence = decoded_preds[0][2]
    
    st.success(f"辨識結果：{food_name_en} (信心度: {confidence:.2%})")
    
    # 呼叫 AI Agent
    with st.spinner('AI 正在撰寫美食報告...'):
        report = generate_food_report(food_name_en)
        st.subheader("🤖 AI Agent 美食報告")
        st.write(report)