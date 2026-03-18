"""
HF Baseline Server — 用原始 Hugging Face transformers 跑 Llama-3.1-8B-Instruct
透過 FastAPI 提供 OpenAI 兼容的 /v1/completions 與 /v1/chat/completions 端點
供 genai-perf 進行 Benchmark 測試

啟動方式:
    python hf_baseline_server.py

測試方式:
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
"""

import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────
# 設定區
# ──────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# 偵測硬體裝置：優先 CUDA -> 其次 Mac MPS -> 最後 CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16 # Mac MPS 穩定支援 float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

# 全域變數（在 lifespan 中初始化）
model = None
tokenizer = None


# ──────────────────────────────────────────────
# FastAPI Lifespan：啟動時載入模型
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"🔄 正在下載並載入模型：{MODEL_ID}...")
    print(f"   裝置：{DEVICE} | 精度：{DTYPE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    model.eval()

    print(f"✅ 模型載入完成！VRAM 佔用：{torch.cuda.memory_allocated() / 1e9:.2f} GB" if DEVICE == "cuda" else "✅ 模型載入完成 (CPU mode)")
    yield
    print("👋 Server 關閉中...")


app = FastAPI(title="HF Baseline LLM Server", lifespan=lifespan)


# ──────────────────────────────────────────────
# Pydantic Models (OpenAI 兼容格式)
# ──────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llama-3.1-8b"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


class CompletionRequest(BaseModel):
    model: str = "llama-3.1-8b"
    prompt: str | list[str]
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)


# ──────────────────────────────────────────────
# 核心推理邏輯
# ──────────────────────────────────────────────
def generate_text(prompt: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    """用最原始的 HF transformers 做推理，並記錄時間"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    end = time.perf_counter()

    # 只取生成的部分（去掉 prompt）
    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    total_tokens = len(generated_ids)
    elapsed = end - start

    return {
        "text": generated_text,
        "prompt_tokens": input_len,
        "completion_tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
    }


# ──────────────────────────────────────────────
# API Endpoints (OpenAI 兼容)
# ──────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI Chat Completions 兼容端點"""
    # 把 messages 轉成 prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    result = generate_text(prompt, request.max_tokens, request.temperature, request.top_p)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI Completions 兼容端點"""
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    result = generate_text(prompt, request.max_tokens, request.temperature, request.top_p)

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": result["text"],
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


@app.get("/v1/models")
async def list_models():
    """模型列表端點（genai-perf 可能需要）"""
    return {
        "object": "list",
        "data": [
            {
                "id": "llama-3.1-8b",
                "object": "model",
                "owned_by": "meta",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


# ──────────────────────────────────────────────
# 啟動入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("   HF Baseline LLM Server")
    print(f"   Model: {MODEL_ID}")
    print(f"   Port:  8000")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
