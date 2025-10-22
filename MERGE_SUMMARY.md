# Requirements文件合并总结

## 📝 合并说明

已将 `ENV.txt` 和原 `requirements.txt` 合并，创建了更完善的依赖管理系统。

## 📦 新的文件结构

### 1. requirements-minimal.txt（新建）✨
**用途**：基础评估功能
**包含**：8个核心依赖
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
pandas>=2.0.0
tqdm>=4.65.0
requests>=2.31.0
swanlab>=0.3.0
```

**适用场景**：
- ✅ 运行 auto_eval.sh/py 脚本
- ✅ 基础模型评估
- ✅ 快速安装（5-10分钟）
- ✅ 节省磁盘空间（~5GB）

### 2. requirements.txt（更新）🔄
**用途**：完整深度学习环境
**包含**：120+个依赖（合并自ENV.txt）

**分类包含**：
- 🔥 深度学习框架：torch, torchvision, torchaudio
- 🤗 HuggingFace生态：transformers, datasets, tokenizers
- 📊 数据处理：pandas, numpy, pyarrow
- ⚡ 训练加速：accelerate, bitsandbytes, peft, deepspeed
- 🚀 推理优化：vllm, xformers, triton, flashinfer
- 📈 监控工具：wandb, swanlab, prometheus, nvitop
- 🔧 工具库：fastapi, uvicorn, ray
- 🎨 可视化：matplotlib, seaborn
- 🧮 数学计算：scipy, scikit-learn, numba

**适用场景**：
- ✅ 完整的训练+推理+评估环境
- ✅ 使用高级优化功能
- ✅ API服务部署
- ✅ 生产环境

### 3. ENV.txt（保留）📋
**用途**：参考用
**包含**：175个原始依赖
**说明**：保留原始环境导出，仅供参考

## 🔄 主要变化

### 合并前
```
requirements.txt  (8个包，版本要求宽松)
ENV.txt          (175个包，精确版本)
```

### 合并后
```
requirements-minimal.txt  (8个包，推荐日常使用) ⭐ 推荐
requirements.txt         (120+个包，完整环境)
ENV.txt                  (175个包，仅参考)
```

## 📊 对比表格

| 特性 | minimal | full (新) | ENV.txt |
|------|---------|----------|---------|
| 包数量 | 8 | 120+ | 175 |
| 版本控制 | 灵活(>=) | 精确(==) | 精确(==) |
| 安装时间 | 5-10分钟 | 20-30分钟 | 30-60分钟 |
| 磁盘空间 | ~5GB | ~15GB | ~20GB |
| 适用场景 | 评估 | 全功能 | 参考 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

## 🎯 使用建议

### 场景1：只需评估模型（90%用户）
```bash
pip install -r requirements-minimal.txt
```

### 场景2：需要训练/推理优化
```bash
pip install -r requirements.txt
```

### 场景3：完全复现原环境
```bash
pip install -r ENV.txt
```

## ✅ 优势

### 1. 分层设计
- **最小依赖**：快速上手，满足基本需求
- **完整依赖**：功能齐全，适合高级用户
- **原始环境**：完整记录，便于复现

### 2. 灵活性
- minimal使用`>=`允许更新
- requirements使用`==`确保稳定
- 用户可根据需求选择

### 3. 文档完善
- [INSTALL.md](INSTALL.md) - 安装指南
- [DEPENDENCIES.md](DEPENDENCIES.md) - 依赖说明
- [README_EVAL.md](README_EVAL.md) - 系统总览

## 🔧 技术细节

### 去重和分类
从ENV.txt提取的175个包中：
- ✅ 保留了所有重要依赖
- ✅ 按功能分类组织
- ✅ 添加了注释说明
- ✅ 去除了重复和冗余

### 版本管理
```python
# requirements-minimal.txt (灵活)
torch>=2.0.0          # 允许升级到2.x任意版本

# requirements.txt (精确)
torch==2.6.0          # 锁定具体版本

# ENV.txt (导出)
torch==2.6.0          # 完全精确
```

## 📚 相关文档

1. [INSTALL.md](INSTALL.md) - 详细安装指南
2. [DEPENDENCIES.md](DEPENDENCIES.md) - 依赖包说明
3. [AUTO_EVAL_README.md](AUTO_EVAL_README.md) - 使用文档
4. [CONDA_QUICK_START.md](CONDA_QUICK_START.md) - Conda指南

## 🎉 总结

通过合并requirements.txt和ENV.txt：

✅ **更清晰** - 分层管理，各取所需
✅ **更灵活** - 多种安装方案
✅ **更快速** - minimal版快速安装
✅ **更完整** - full版功能齐全
✅ **更易用** - 文档完善，示例丰富

### 推荐工作流

```bash
# 1. 快速开始
pip install -r requirements-minimal.txt
./auto_eval.sh

# 2. 如需高级功能，升级到完整版
pip install -r requirements.txt

# 3. 验证环境
python3 check_env.py
```

## 🔗 快速链接

- 开始使用：`./test_setup.sh`
- 运行评估：`./auto_eval.sh`
- 检查环境：`python3 check_env.py`
- 查看文档：[README_EVAL.md](README_EVAL.md)
