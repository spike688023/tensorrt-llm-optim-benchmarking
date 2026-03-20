# LLM Inference Benchmarking: Hugging Face vs. TensorRT-LLM

本專案旨在提供一個標準化的 LLM 推理性能測試框架，比較 Hugging Face 原始模型與 NVIDIA TensorRT-LLM 優化後的效能差異。

## 🚀 快速開始

我們將在 RunPod 進行測試。為了完美支援後續的 TensorRT-LLM 編譯與正版 `genai-perf`，**強烈建議直接使用 NVIDIA 官方的 Triton 映像檔開機**。

### 1. 租用 GPU (RunPod) 並選擇正確 Image
*   **連結：** [RunPod GPU Cloud](https://www.runpod.io/console/gpu-browse)
*   **建議規格：** NVIDIA **L4 (24GB)** 或 **A10 (24GB)**
*   **【重點】Template / Image Name 必須填入：** 
    `nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3`
*   **Container Disk (容器容量)：** 建議拉大至 **50GB+**，以容納龐大的 TensorRT 引擎檔。

### 2. 環境設定與一鍵測試 (Hugging Face Baseline) 
登入機器後（可使用 Web Terminal），請依序執行以下指令：

```bash
# A. 下載專案
git clone https://github.com/spike688023/tensorrt-llm-optim-benchmarking.git
cd tensorrt-llm-optim-benchmarking

# B. 設定 Hugging Face Token (用於下載 Llama-3.1-8B)
echo "HF_TOKEN=hf_your_token_here" > .env

# C. 執行一鍵自動化測試 (這會產出 Baseline 基準成績單)
chmod +x auto_benchmark_hf.sh
./auto_benchmark_hf.sh
```

---

## 📊 關鍵量測指標 (Metrics)

本工具會自動抓取以下 LLM 推理核心指標：

1.  **TTFT (Time To First Token)**: 首字延遲 (ms)
2.  **ITL (Inter-Token Latency)**: 字間平均延遲 (ms)
3.  **Throughput (TPS)**: 每秒生成 Token 數量

---

## 🛠️ 專案檔案說明

*   `auto_benchmark_hf.sh`: **一鍵自動化測試**，包含環境安裝、啟動伺服器與跑測試。
*   `hf_baseline_server.py`: 使用原始 Hugging Face 跑的 FastAPI Server (OpenAI 相容)。
*   `run_benchmarks.sh`: 呼叫原廠 `genai-perf` (透過 `--service-kind openai`) 進行不同 Concurrency 壓測。
*   `summarize_results.py`: 讀取 CSV 數據並輸出總結表格。
*   `tensorrt-llm-optimization.md`: TensorRT-LLM 原理與部落格教學。

---

## 🔒 隱私與安全
請確保你的 `.env` 被加入 `.gitignore` 且不要將 HF Token 推送到公共倉庫。本專案預設已將 `.env` 排除。

---

**Next Steps:** 在跑完上述的 Baseline 之後，我們將進入 TensorRT-LLM 編譯階段，並進行效能對比。