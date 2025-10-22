# 安装指南

## 📦 依赖文件说明

本项目提供三种安装方式，对应三个依赖文件：

| 文件 | 用途 | 包数量 | 推荐场景 |
|------|------|--------|---------|
| **requirements-minimal.txt** | 基础评估 | ~8个 | ✅ 仅运行auto_eval脚本 |
| **requirements.txt** | 完整环境 | ~120个 | 训练+推理+评估 |
| **ENV.txt** | 原始导出 | ~175个 | 参考用 |

## 🚀 快速安装（推荐）

### 方式1：自动安装（最简单）
```bash
# 脚本会自动安装基础依赖
./auto_eval.sh
```

### 方式2：最小依赖安装
```bash
# 仅安装评估必需的包（推荐）
pip install -r requirements-minimal.txt
```

### 方式3：完整环境安装
```bash
# 安装所有包（包含训练、推理等高级功能）
pip install -r requirements.txt
```

## 📋 分步安装指南

### 步骤1：创建Conda环境（推荐）
```bash
# 创建新环境
conda create -n eval_env python=3.10 -y

# 激活环境
conda activate eval_env
```

### 步骤2：安装依赖

#### 选项A：最小依赖（仅评估）
```bash
pip install -r requirements-minimal.txt
```

包含：
- torch (深度学习框架)
- transformers (HuggingFace模型)
- datasets (数据集加载)
- pandas (数据处理)
- tqdm (进度条)
- requests (HTTP请求)
- swanlab (实验跟踪)

#### 选项B：完整依赖（全功能）
```bash
pip install -r requirements.txt
```

额外包含：
- 训练加速：accelerate, bitsandbytes, peft, deepspeed
- 推理优化：vllm, xformers, triton
- 监控工具：wandb, nvitop
- API服务：fastapi, uvicorn
- 更多...

### 步骤3：验证安装
```bash
# 运行环境检查
python3 check_env.py

# 或运行完整测试
./test_setup.sh
```

## 🎯 按场景选择

### 场景1：仅评估模型（推荐新手）
```bash
conda create -n eval_env python=3.10 -y
conda activate eval_env
pip install -r requirements-minimal.txt
./auto_eval.sh
```

### 场景2：完整深度学习环境
```bash
conda create -n dl_env python=3.10 -y
conda activate dl_env
pip install -r requirements.txt
```

### 场景3：已有环境，仅添加评估工具
```bash
conda activate your_existing_env
pip install swanlab datasets tqdm requests
```

## 🔧 依赖版本说明

### requirements-minimal.txt（灵活版本）
使用 `>=` 允许安装更新版本：
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
```

### requirements.txt（固定版本）
使用 `==` 锁定特定版本：
```
torch==2.6.0
transformers==4.50.1
datasets==3.4.1
```

## 🐛 常见问题

### Q1: 安装torch时CUDA版本不匹配
```bash
# CPU版本
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q2: 国内网络慢
```bash
# 使用清华镜像
pip install -r requirements-minimal.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像
pip install -r requirements-minimal.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### Q3: 权限错误
```bash
# 使用用户安装
pip install --user -r requirements-minimal.txt
```

### Q4: 某些包安装失败
```bash
# 跳过失败的包继续安装
pip install -r requirements.txt --ignore-errors

# 或逐个安装必需的包
pip install torch transformers datasets pandas tqdm requests swanlab
```

### Q5: 想要特定版本的包
```bash
# 修改requirements文件或直接指定
pip install torch==2.0.0 transformers==4.30.0
```

## 📊 安装时间估计

| 安装方式 | 预计时间 | 磁盘空间 |
|---------|---------|---------|
| requirements-minimal.txt | 5-10分钟 | ~5GB |
| requirements.txt | 20-30分钟 | ~15GB |
| ENV.txt | 30-60分钟 | ~20GB |

*注：首次安装需要下载，后续使用缓存会快很多*

## ✅ 验证安装

### 快速验证
```bash
python3 -c "import torch, transformers, datasets; print('✅ 基础包已安装')"
```

### 完整验证
```bash
# 运行环境检查脚本
python3 check_env.py
```

期望输出：
```
✅ torch           2.6.0
✅ transformers    4.50.1
✅ datasets        3.4.1
✅ pandas          2.2.3
✅ tqdm            4.67.1
✅ requests        2.32.3
✅ swanlab         0.7.2-dev

✅ 所有依赖已正确安装！
```

### 测试CUDA
```bash
python3 -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

## 🔄 更新依赖

### 更新所有包到最新版本
```bash
pip install --upgrade -r requirements-minimal.txt
```

### 更新单个包
```bash
pip install --upgrade transformers
```

### 重新安装
```bash
pip install --force-reinstall -r requirements-minimal.txt
```

## 🗑️ 卸载

### 卸载Conda环境
```bash
conda deactivate
conda env remove -n eval_env
```

### 仅卸载特定包
```bash
pip uninstall torch transformers datasets -y
```

## 📚 更多帮助

- 完整文档：[AUTO_EVAL_README.md](AUTO_EVAL_README.md)
- 依赖说明：[DEPENDENCIES.md](DEPENDENCIES.md)
- Conda指南：[CONDA_QUICK_START.md](CONDA_QUICK_START.md)

## 💡 最佳实践

1. ✅ **使用Conda环境** - 避免污染系统Python
2. ✅ **先装minimal** - 够用就行，需要时再装完整版
3. ✅ **定期更新** - 保持依赖包最新
4. ✅ **验证安装** - 运行check_env.py确认
5. ✅ **记录环境** - 导出requirements便于复现

```bash
# 导出当前环境
pip freeze > my_environment.txt

# 在其他机器复现
pip install -r my_environment.txt
```
