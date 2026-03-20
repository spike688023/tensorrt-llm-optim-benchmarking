#!/bin/bash

# ──────────────────────────────────────────────
# HF Baseline 一鍵自動化測試腳本
# ──────────────────────────────────────────────

# 1. 載入與檢查 Token
if [ -f .env ]; then
    echo "📂 載入 .env 變數..."
    set -a; source .env; set +a
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ 錯誤: 請在 .env 內設定 HF_TOKEN"
    exit 1
fi

# 2. 自動安裝必要套件 (如果還沒裝)
echo "📦 正在檢查套件環境..."
pip3 install --upgrade pip
pip3 install -q "python-dotenv" "fastapi" "uvicorn" "torch" "transformers" "accelerate" "tritonclient[all]" "huggingface_hub"

# 3. 執行登入
echo "🔑 正在驗證 Hugging Face Token..."
python3 -m huggingface_hub.commands.user login --token $HF_TOKEN --add-to-git-credential

# 4. 在背景啟動 Server
echo "🚀 正在啟動 HF Baseline Server (背景執行)..."
python3 hf_baseline_server.py > server.log 2>&1 &
SERVER_PID=$!

# 5. 等待 Server 準備好 (Polling /health endpoint)
echo "⏳ 等待 Server 初始化與載入模型 (這可能需要幾分鐘，取決於下載速度)..."
MAX_RETRIES=60
RETRY_COUNT=0
until $(curl --output /dev/null --silent --head --fail http://localhost:8000/health); do
    printf '.'
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "❌ 超時失敗：Server 在 5 分鐘內沒有反應，請檢查 server.log"
        kill $SERVER_PID
        exit 1
    fi
done
echo -e "\n✅ Server 已準備就緒！"

# 6. 執行 Benchmark
echo "📊 開始執行性能測試..."
chmod +x run_benchmarks.sh
./run_benchmarks.sh

# 7. 測試結束，關閉 Server
echo "🛑 測試結束，正在關閉 Server..."
kill $SERVER_PID

# 8. 顯示最終總結報告
echo "📈 正在生成結果總結..."
python3 summarize_results.py

echo "================================================"
echo "🎉 所有測試已完成！詳細數據請參考 ./results 目錄"
echo "================================================"
