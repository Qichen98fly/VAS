# 1) 安装 CUDA toolkit（带 nvcc）。若你 torch 是 cu121，装 12.1 最稳
conda install -y -c nvidia cuda-toolkit=12.1

# 2) 设置环境变量（当前 shell 有效；写到 ~/.bashrc 可永久）
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# 3) 检查
nvcc --version    # 能输出版本说明 nvcc 可用
python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'py', sys.version.split()[0])"

# 4) 可选：加速编译
export USE_NINJA=1
export MAX_JOBS=$(nproc)

# 5) 编译安装（2.6.3 对 torch 2.4 / cu121 常见可配）
pip install flash-attn==2.6.3 --no-build-isolation
