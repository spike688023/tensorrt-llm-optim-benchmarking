#!/bin/bash

# ──────────────────────────────────────────────
# genai-perf 測試腳本 (針對 HF Baseline Server)
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# 1. 檢查與載入環境變數 (.env)
# ──────────────────────────────────────────────
if [ -f .env ]; then
    echo "📂 偵測到 .env 檔案，正在載入變數..."
    set -a; source .env; set +a
fi

# Removed HF Token login as it is handled globally by auto_benchmark_hf.sh

export URL="localhost:8000"
export MODEL_NAME="llama-3.1-8b"

echo "================================================"
echo "🚀 開始執行 LLM 性能測試 (Hugging Face Baseline)"
echo "Target Model: $MODEL_NAME"
echo "Target URL:   $URL"
echo "================================================"

# 檢查 custom_bench.py 是否存在
if [ ! -f "custom_bench.py" ]; then
    echo "❌ 找不到自訂跑分腳本 custom_bench.py！"
    exit 1
fi

# 建立輸出目錄
mkdir -p ./results

# ----------------------------------------------
# 測試 1：單次請求 (Latency 導向)
# ----------------------------------------------
echo "--- 正在測試：單人低延遲模式 (Concurrency=1) ---"
python3 custom_bench.py --concurrency 1 --num-prompts 5 --output-dir ./results/concurrency_1

# ----------------------------------------------
# 測試 2：多併發請求 (Throughput 導向)
# ----------------------------------------------
echo "--- 正在測試：四人併發抗壓模式 (Concurrency=4) ---"
python3 custom_bench.py --concurrency 4 --num-prompts 20 --output-dir ./results/concurrency_4

echo "================================================"
echo "✅ 測試完成！請查看 ./results 目錄下的報告。"
echo "================================================"
