import os
import pandas as pd
from glob import glob

def extract_metrics(result_path):
    """從 genai-perf 的 csv 產出中撈取關鍵指標"""
    # 尋找目錄下的 csv 檔案（通常是像是 profile_export_... 或 metrics.csv）
    csv_files = glob(os.path.join(result_path, "*.csv"))
    if not csv_files:
        return None
    
    # 讀取數據 (假設 genai-perf 輸出的格式)
    # 我們通常看平均值 (Avg) 或 P90 (90% 使用者體驗)
    try:
        # 有時候會有兩三份 csv，我們找包含 summary 字樣的
        summary_file = next((f for f in csv_files if "summary" in f.lower()), csv_files[0])
        df = pd.read_csv(summary_file)
        
        # 轉換成 dict 方便讀取 (這部分會根據 genai-perf 版本微調)
        # 典型的指標名稱：'time_to_first_token', 'inter_token_latency', 'output_token_throughput'
        metrics = {}
        for _, row in df.iterrows():
            stat = row.get('Statistic', row.get('metric', ''))
            val = row.get('Value', row.get('value', 0))
            metrics[stat] = val
        return metrics
    except Exception as e:
        print(f"⚠️ 解析 {result_path} 失敗: {e}")
        return None

def print_table(results):
    print("\n" + "="*80)
    print(f"{'Test Scenario':<20} | {'TTFT (ms)':<15} | {'ITL (ms)':<15} | {'TPS'}")
    print("-" * 80)
    for scenario, m in results.items():
        if m:
            # 這裡的 key 名稱要對應 genai-perf 的輸出
            # 注意：某些版本單位是 ns，這裡我假設它會輸出 ms 或我們自己轉
            ttft = m.get('avg_time_to_first_token', 'N/A')
            itl  = m.get('avg_inter_token_latency', 'N/A')
            tps  = m.get('output_token_throughput', 'N/A')
            print(f"{scenario:<20} | {ttft:<15.2f} | {itl:<15.2f} | {tps:.2f}")
        else:
            print(f"{scenario:<20} | {'No Data':<15} | {'No Data':<15} | {'No Data'}")
    print("="*80 + "\n")

if __name__ == "__main__":
    results = {}
    
    # 撈取剛才腳本跑出的兩個目錄
    results["Single-user (C=1)"] = extract_metrics("./results/concurrency_1")
    results["Multi-user (C=4)"]  = extract_metrics("./results/concurrency_4")
    
    print_table(results)
