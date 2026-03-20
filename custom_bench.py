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
        with requests.post(URL, json=payload, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if line and line.startswith(b"data:"):
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now - start_time
                    token_times.append(now)
                    
        total_time = time.time() - start_time
        num_tokens = len(token_times)
        
        if first_token_time is not None:
            ttfts.append(first_token_time * 1000) # ms
        if len(token_times) > 1:
            # ITL: 每個 token 之間的平均延遲
            itl = np.mean([token_times[i] - token_times[i-1] for i in range(1, len(token_times))])
            itls.append(itl * 1000) # ms
            # 單一連線吞吐量 (Tokens/Sec)
            throughputs.append(num_tokens / total_time) 
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
