# Optimizing Inference on Large Language Models with NVIDIA TensorRT-LLM, Now Publicly Available

**Authors:** [Neal Vaidya](https://developer.nvidia.com/blog/author/nealv/), [Fred Oh](https://developer.nvidia.com/blog/author/fredoh/), [Nick Comly](https://developer.nvidia.com/blog/author/nickcomly/)

---

Today, NVIDIA announces the public release of TensorRT-LLM to accelerate and optimize inference performance for the latest LLMs on NVIDIA GPUs. This open-source library is now available for free on the [/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0) GitHub repo and as part of the [NVIDIA NeMo framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/).

Large language models (LLMs) have revolutionized the field of artificial intelligence and created entirely new ways of interacting with the digital world. But, as organizations and application developers around the world look to incorporate LLMs into their work, some of the challenges with running these models become apparent.

Put simply, LLMs are large. That fact can make them expensive and slow to run without the right techniques.

Many optimization techniques have risen to deal with this, from model optimizations like [kernel fusion](https://arxiv.org/abs/2307.08691) and [quantization](https://nvidia.github.io/TensorRT-LLM/precision.html) to runtime optimizations like C++ implementations, KV caching, [continuous in-flight batching](https://www.usenix.org/conference/osdi22/presentation/yu), and [paged attention](https://arxiv.org/pdf/2309.06180.pdf). It can be difficult to decide which of these are right for your use case, and to navigate the interactions between these techniques and their sometimes-incompatible implementations.

That’s why [NVIDIA introduced TensorRT-LLM](https://developer.nvidia.com/blog/nvidia-tensorrt-llm-supercharges-large-language-model-inference-on-nvidia-h100-gpus/), a comprehensive library for compiling and optimizing LLMs for inference. TensorRT-LLM incorporates all of those optimizations and more while providing an intuitive Python API for defining and building new models.

The TensorRT-LLM open-source library accelerates inference performance on the latest LLMs on NVIDIA GPUs. It is used as the optimization backbone for LLM inference in NVIDIA NeMo, an end-to-end framework to build, customize, and deploy generative AI applications into production. NeMo provides complete containers, including TensorRT-LLM and NVIDIA Triton, for generative AI deployments.

[TensorRT-LLM is also now available for native Windows](https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/) as a beta release. Application developers and AI enthusiasts can now benefit from accelerated LLMs running locally on PCs and Workstations powered by NVIDIA RTX and NVIDIA GeForce RTX GPUs.

TensorRT-LLM wraps TensorRT’s deep learning compiler and includes the latest optimized kernels made for cutting-edge implementations of [FlashAttention](https://arxiv.org/abs/2307.08691) and masked multi-head attention (MHA) for LLM execution.

TensorRT-LLM also consists of pre– and post-processing steps and multi-GPU/multi-node communication primitives in a simple, open-source Python API for groundbreaking LLM inference performance on GPUs.

### Highlights of TensorRT-LLM:
- Support for LLMs such as Llama 1 and 2, ChatGLM, Falcon, MPT, Baichuan, and Starcoder
- In-flight batching and paged attention
- Multi-GPU multi-node (MGMN) inference
- NVIDIA Hopper transformer engine with FP8
- Support for NVIDIA Ampere architecture, NVIDIA Ada Lovelace architecture, and NVIDIA Hopper GPUs
- Native Windows support (beta)

Over the past 2 years, NVIDIA has been working closely with leading LLM companies, including Anyscale, Baichuan, Cohere, Deci, Grammarly, Meta, Mistral AI, MosaicML (now part of Databricks), OctoML, Perplexity AI, Tabnine, Together.ai, Zhipu, and many others to accelerate and optimize LLM inference.

---

## Getting started with installation

Here are the steps to get started with TensorRT-LLM:
1. Retrieving the model weights
2. Installing the TensorRT-LLM library
3. Compiling the model
4. Running the model
5. Deploying with Triton Inference Server
6. Sending requests

### 1. Retrieving the model weights
TensorRT-LLM is a library for LLM inference, and so to use it, you must supply a set of trained weights. You can either use your own model weights trained in a framework like [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/) or pull a set of pretrained weights from repositories like the [HuggingFace Hub](https://huggingface.co/docs/hub/en/index).

The commands in this post automatically pull the weights and tokenizer files for the [chat-tuned variant of the 7B parameter Llama 2 model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) from the HuggingFace Hub.

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

### 2. Installing the TensorRT-LLM Library
Launch a Docker container and install the TensorRT-LLM library.

```bash
# Obtain and start the basic docker image environment.
docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install dependencies (including git for cloning samples)
apt-get update && apt-get -y install git python3.10 python3-pip openmpi-bin libopenmpi-dev

# Clone the TensorRT-LLM repository to get example scripts like convert_checkpoint.py
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Install the latest preview version of the library
pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Check installation
python3 -c "import tensorrt_llm"
```

### 3. Compiling the model
The next step is to compile the model into a TensorRT engine. You need the model weights and a model definition written in the TensorRT-LLM Python API.

```bash
# Log in to huggingface-cli
huggingface-cli login --token *****

# Build the LLaMA 7B model using a single GPU and BF16.
python convert_checkpoint.py --model_dir ./meta-llama/Llama-2-7b-chat-hf \
                             --output_dir ./tllm_checkpoint_1gpu_bf16 \
                             --dtype bfloat16

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_bf16 \
            --output_dir ./tmp/llama/7B/trt_engines/bf16/1-gpu \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16
```

The TensorRT compiler sweeps through the graph to choose the best kernel for each operation and identifies patterns for layer fusion (like FlashAttention).

**Build Outputs in `/tmp/llama/7B/trt_engines/bf16/1-gpu`:**
- `Llama_float16_tp1_rank0.engine`: The executable graph with model weights embedded.
- `config.json`: Model structure and plugin information.
- `model.cache`: Caches timing information for quicker successive builds.

### 4. Running the model
To run the model locally, execute the following command:

```bash
python3 examples/llama/run.py --engine_dir=./tmp/llama/7B/trt_engines/bf16/1-gpu \
                              --max_output_len 100 \
                              --tokenizer_dir meta-llama/Llama-2-7b-chat-hf \
                              --input_text "How do I count to nine in French?"
```

### 5. Deploying with Triton Inference Server
Beyond local execution, you can use [NVIDIA Triton Inference Server](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/) for production-ready deployment.

Setup the model repository:
```bash
# Clone the backend
git clone -b release/0.8.0 https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
cp ../TensorRT-LLM/examples/llama/out/* all_models/inflight_batcher_llm/tensorrt_llm/1/
```

Launch the Triton server:
```bash
docker run -it --rm --gpus all --network host --shm-size=1g \
       -v $(pwd)/all_models:/all_models \
       -v $(pwd)/scripts:/opt/scripts \
       nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3

# Inside container:
huggingface-cli login --token *****
pip install sentencepiece protobuf
python /opt/scripts/launch_triton_server.py --model_repo /all_models/inflight_batcher_llm --world_size 1
```

### 6. Sending requests
Interact via the `generate` endpoint:

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate -d \
'{
  "text_input": "How do I count to nine in French?",
  "parameters": {
    "max_tokens": 100,
    "bad_words": [""],
    "stop_words": [""]
  }
}'
```

---

## Conclusion

TensorRT-LLM and Triton Inference Server provide an indispensable toolkit for optimizing, deploying, and running LLMs efficiently. With the release as an open-source library, it’s easier than ever to harness the potential of these models.

**Resources:**
- [NVIDIA/TensorRT-LLM GitHub Repo](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA NeMo Framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/)
- [Benchmarks and Documentation](https://nvidia.github.io/TensorRT-LLM/index.html)
