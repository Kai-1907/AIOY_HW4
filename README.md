### 🥗 食物辨識與 AIGC 智能營養助手 (Food AI Agent)
本專案為 Taica AIGC 課程 專題實作，旨在展示如何將「深度學習影像辨識」與「大語言模型 (LLM)」結合。透過 CNN 模型辨識上傳的食物影像，並由 AI Agent 自動生成營養分析與美食文案。

### 🚀 專案亮點

深度學習整合：利用 MobileNetV2 進行高效能食物影像分類。

AIGC 延伸應用：串接 Google Gemini API，根據辨識結果進行文本創作。

雲端部署優化：針對 Streamlit Cloud 無 GPU 環境，採用 tensorflow-cpu 與模型量化技術解決部署問題。

安全性設計：使用 Streamlit Secrets 管理 API Key，避免機密流出。

### 🛠️ 技術棧 (Tech Stack)
前端介面: Streamlit

深度學習框架: TensorFlow / Keras (MobileNetV2)

AIGC 引擎: Google Gemini Pro API

部署平台: Streamlit Cloud / GitHub

### 📂 檔案結構
``` Plaintext

.
├── app.py              # 系統主程式 (包含 DL 推論與 Agent 邏輯)
├── requirements.txt    # 預設套件清單 (已優化為 CPU 版本)
└── README.md           # 專案說明文件

```
### 📝 實作步驟

1. 影像辨識 (Deep Learning)
系統使用在 ImageNet 上預訓練的 MobileNetV2 模型。選擇此模型的原因是其參數精簡，適合在不具備專用 GPU 的 Streamlit Cloud 環境下運行。

2. AIGC Agent 開發
當模型辨識出食物標籤（如：pizza, sushi）後，該標籤會作為輸入傳送至 Gemini API。

Prompt 策略：設定 Agent 角色為「專業營養師與美食評論家」。

輸出內容：包含食物介紹、預估熱量及一段社群媒體文案。

3. 部署 (Deployment)
GPU Issue 解決方案：在 requirements.txt 中使用 tensorflow-cpu 替代完整版 TensorFlow，減少部署時的記憶體佔用。

Vercel 參考：本專案亦參考了 Vercel 的部署邏輯進行環境變數配置。

⚙️ 如何在本機執行
複製倉庫:

```Bash

git clone https://github.com/你的用戶名/你的專案名.git
```
安裝依賴:

```Bash

pip install -r requirements.txt
```
設定 API Key: 在專案根目錄建立 .streamlit/secrets.toml 並填入：

```Ini, TOML

GEMINI_API_KEY = "你的_API_KEY"
```
啟動 App:

```Bash

streamlit run app.py
```
### 📊 專題心得
透過本次 Taica AIGC 課程實作，我深入理解了如何將模型從本地環境推向生產環境（Production）。特別是在處理 Streamlit GPU 限制時，學會了如何優化模型選擇與環境配置。此外，將 CNN 與 LLM 結合的過程，讓我看見了 AIGC 如何為傳統人工智慧應用增值。

🔗 相關連結
Streamlit App: [點此開啟你的 App 連結]

Taica 課程官網: https://taicatw.net/fall-114/