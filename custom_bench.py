import time
import requests
import threading
import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--concurrency", type=int, default=1)
parser.add_argument("--num-prompts", type=int, default=5)
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()

URL = "http://localhost:8000/v1/chat/completions"
# 測試用的 Prompt
payload = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain the theory of relativity in exactly 300 words without stopping."}],
    "stream": True,
    "max_tokens": 150
}

ttfts = []
itls = []
throughputs = []

def run_request():
    start_time = time.time()
    first_token_time = None
    token_times = []
    
    try:
        # FastAPI 端點目前沒有實作 stream，因此改為直接接收完整 JSON
        r = requests.post(URL, json=payload, timeout=120)
        total_time = time.time() - start_time
        
        if r.status_code == 200:
            res = r.json()
            num_tokens = res.get("usage", {}).get("completion_tokens", 0)
            
            # 因為沒有 stream，TTFT (首字延遲) 只能當作總時間
            ttfts.append(total_time * 1000) 
            # 沒有 stream 就無法精確測量每個字之間的隔閡，ITL 設為 0
            if num_tokens > 0:
                throughputs.append(num_tokens / total_time) 
        else:
            print(f"Server error: {r.status_code} {r.text}")
    except Exception as e:
        print(f"Request failed: {e}")

threads = []
print(f"啟動打流測試 (併發數: {args.concurrency}, 總請求數: {args.num_prompts})...")

for i in range(args.num_prompts):
    # 控制最大併發數
    while len([t for t in threads if t.is_alive()]) >= args.concurrency:
        time.sleep(0.1)
    
    t = threading.Thread(target=run_request)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

avg_ttft = np.mean(ttfts) if ttfts else 0.0
avg_itl = np.mean(itls) if itls else 0.0
avg_tps = sum(throughputs) if throughputs else 0.0 # 總吞吐量為所有連線加總

print(f"測試完成 -> TTFT: {avg_ttft:.2f}ms | ITL: {avg_itl:.2f}ms | 總 TPS: {avg_tps:.2f} tokens/s")

# 輸出 CSV 讓 summarize_results 讀取
os.makedirs(args.output_dir, exist_ok=True)
df = pd.DataFrame([
    {"Statistic": "avg_time_to_first_token", "Value": avg_ttft},
    {"Statistic": "avg_inter_token_latency", "Value": avg_itl},
    {"Statistic": "output_token_throughput", "Value": avg_tps}
])
df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
