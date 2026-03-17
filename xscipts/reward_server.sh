 #!/bin/bash
# VerIF Verifier 跨机器部署脚本 - vLLM 版本

# ============ Conda 环境配置 ============
# 激活 conda 基础环境
# source /mnt/public/users/tuokaiwen/miniconda3/bin/activate
# 激活 verl 环境
# conda activate verl

# export http_proxy=http://jdtcom:709a64b73eb3@10.119.176.202:3128
# export https_proxy=http://jdtcom:709a64b73eb3@10.119.176.202:3128

# ============ HuggingFace 缓存配置 ============
export HF_HOME=/mnt/public/users/zhangjinghao/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub


# 打印验证缓存路径
echo "HuggingFace 缓存配置:"
echo "  HF_HOME: $HF_HOME"
echo "  离线模式: HF_HUB_OFFLINE=1 ✓"
echo ""

# ============ 模型配置 ============
# ✅ 使用本地已下载的模型路径（推荐，避免网络连接和重复下载）
MODEL_PATH="/mnt/public/users/zhangjinghao/models/Qwen3-30B-A3B-Instruct-2507"

# 备选：使用 HuggingFace model ID（需要网络连接）
# MODEL_PATH="Qwen/QwQ-32B"

HOST="0.0.0.0"  # 监听所有网络接口，允许跨机器访问
PORT=8000

# ============ ============
TENSOR_PARALLEL_SIZE=1  
GPU_MEMORY_UTILIZATION=0.90  #
MAX_MODEL_LEN=16384  # 
MAX_NUM_SEQS=512  # 
ENABLE_CHUNKED_PREFILL=true  # 启用分块预填充，更好地处理长序列

# 打印配置信息
echo "=========================================="
echo "  启动 Verifier API 服务 (vLLM + 4×H200)"
echo "=========================================="
echo "模型路径: ${MODEL_PATH}"
echo "监听地址: ${HOST}:${PORT}"
echo "GPU 配置: ${TENSOR_PARALLEL_SIZE} × H200 (张量并行)"
echo "显存利用: ${GPU_MEMORY_UTILIZATION} (每卡约 134GB)"
echo "最大长度: ${MAX_MODEL_LEN} tokens"
echo "最大并发: ${MAX_NUM_SEQS} 个请求 (匹配GRPO单step需求)"
echo "分块预填充: ${ENABLE_CHUNKED_PREFILL}"
echo ""
echo "本机 IP 地址："
hostname -I | awk '{print $1}'
echo ""
echo "  • 单step可同时处理 ${MAX_NUM_SEQS} 个请求 (无需排队)"
echo "  • 单次评分延迟: ~0.5-1s (含网络+推理)"
echo ""
echo "服务启动后可通过以下地址访问："
echo "  本地: http://localhost:${PORT}/v1"
echo "  远程: http://$(hostname -I | awk '{print $1}'):${PORT}/v1"
echo "=========================================="
echo ""

# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --host ${HOST} \
    --port ${PORT} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --enable-chunked-prefill \
    --served-model-name "qwen3"