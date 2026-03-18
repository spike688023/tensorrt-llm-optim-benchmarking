from huggingface_hub import model_info

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" # 或是 "Qwen/Qwen2.5-7B-Instruct"

try:
    info = model_info(MODEL_ID)
    print(f"✅ 權限確認！模型名稱：{info.id}")
except Exception as e:
    print(f"❌ 權限失敗或需要核准：{e}")
