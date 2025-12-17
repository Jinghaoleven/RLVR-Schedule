pip install vllm==0.11.0
pip install "numpy<2.0.0" "pyarrow>=15.0.0" pandas
pip install codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler
pip install pytest py-spy pyext pre-commit ruff tensorboard 
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install flashinfer-python==0.2.2

pip install tensordict hydra-core torchdata codetiming datasets
pip install uvloop==0.21.0