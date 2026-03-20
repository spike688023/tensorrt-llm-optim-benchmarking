import os
from huggingface_hub import model_info, login
from dotenv import load_dotenv

# 載入 .env
load_dotenv()
token = os.getenv("HF_TOKEN")

if token:
    login(token=token)
    print("✅ 已從 .env 載入 Token！")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

try:
    info = model_info(MODEL_ID)
    print(f"✅ 權限驗證成功！模型：{info.id}")
except Exception as e:
    print(f"❌ 驗證失敗：{e}")
