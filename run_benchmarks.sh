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

# 檢查 genai-perf 是否存在
if ! command -v genai-perf &> /dev/null; then
    echo "❌ 找不到 genai-perf 指令！"
    echo "請確認是否在 NVIDIA Triton 專屬 Container 內。"
    exit 1
fi

# 建立輸出目錄
mkdir -p ./results

# ----------------------------------------------
# 測試 1：單次請求 (Latency 導向)
# ----------------------------------------------
echo "--- 正在測試：單人低延遲模式 (Concurrency=1) ---"
genai-perf \
    --model $MODEL_NAME \
    --service-kind openai \
    --url $URL \
    --endpoint v1/chat/completions \
    --concurrency 1 \
    --num-prompts 5 \
    --random-seed 42 \
    --output-dir ./results/concurrency_1

# ----------------------------------------------
# 測試 2：多併發請求 (Throughput 導向)
# ----------------------------------------------
echo "--- 正在測試：四人併發抗壓模式 (Concurrency=4) ---"
genai-perf \
    --model $MODEL_NAME \
    --service-kind openai \
    --url $URL \
    --endpoint v1/chat/completions \
    --concurrency 4 \
    --num-prompts 20 \
    --random-seed 42 \
    --output-dir ./results/concurrency_4

echo "================================================"
echo "✅ 測試完成！請查看 ./results 目錄下的報告。"
echo "================================================"
