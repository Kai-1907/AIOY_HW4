### 🥗 AI 食物辨識與智能營養評論 Agent (Taica AIGC 專題)
### 📌 專案簡介 (ABSTRACT)
本專案實作一個結合 深度學習 (Deep Learning) 與 生成式 AI (AIGC) 的智能食物分析系統。系統核心採用 MobileNetV2 卷積神經網絡進行影像辨識，並延伸開發「多模態 Agent」功能。當使用者上傳食物照片後，系統不僅能準確分類，還能透過 Google Gemini API 扮演專業營養師，自動生成營養分析與社群媒體美食文案，展示了從辨識（Recognition）到生成（Generation）的完整 AI 應用鏈。

### 🚀 核心功能與技術實作
影像辨識引擎 (DL)：使用預訓練的 MobileNetV2 模型，針對上傳影像進行特徵提取與 1000 種以上的物體分類。

智能文案 Agent (AIGC)：串接 Gemini-Pro 模型，根據辨識標籤進行脈絡化創作，生成包含營養資訊與幽默風格的推薦文字。

部署優化 (GPU Issue)：針對 Streamlit Cloud 的硬體限制，本專案放棄使用昂貴的 GPU 資源，改採用 tensorflow-cpu 進行推論，並透過 @st.cache_resource 優化模型加載速度。

### 🛠️ 技術棧 (Tech Stack)
核心語言: Python 3.11

深度學習: TensorFlow 2.x (CPU 部署版本)

網頁框架: Streamlit

生成式 AI: Google Generative AI (Gemini API)

雲端平台: GitHub + Streamlit Cloud

### 📂 專案檔案結構
```Plaintext

.
├── .streamlit/
│   └── secrets.toml    # 本地端 API Key 管理 (已加入 .gitignore)
├── app.py              # 整合 DL 與 AIGC 的主程式
├── requirements.txt    # 雲端部署套件清單 (優化 GPU 相容性)
├── ABSTRACT.md         # 專題摘要 300 字
└── README.md           # 本說明文件
```
### 🔧 安裝與部署指南
1. 解決 GPU 與環境問題
在部署至 Streamlit Cloud 時，請確保 requirements.txt 使用以下配置以避免 GPU 報錯：

```Plaintext

streamlit
tensorflow-cpu
pillow
numpy
google-generativeai
```
2. 安全性設定 (Secrets)
本專案不將 API Key 上傳至 GitHub。請在 Streamlit Cloud 的後台 Settings -> Secrets 中填入：

```Ini, TOML

GEMINI_API_KEY = "你的_Gemini_API_Key"
```
### 💬 Agent 開發過程對話紀錄 (摘要)
問題: 如何在無 GPU 環境部署大型深度學習模型？

對策: 經由與 AI 夥伴討論，決定採用 MobileNetV2 輕量化模型，並將 tensorflow 更換為 tensorflow-cpu 以節省 80% 的安裝體積。

問題: 如何提升 Agent 輸出的專業度？

對策: 透過 Prompt Engineering，為 Gemini 設定「專業美食評論家」的角色，並強制要求輸出包含營養成分與 IG 文案。

🔗 連結
GitHub 倉庫: [請在此填入你的 GitHub 網址]

Streamlit Demo: [請在此填入你的 Streamlit.app 網址]