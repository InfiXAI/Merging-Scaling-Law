#!/bin/bash
# 测试脚本 - 验证自动评估脚本的设置

echo "=================================="
echo "测试自动评估脚本设置"
echo "=================================="
echo ""

# 1. 检查脚本文件
echo "1️⃣  检查脚本文件..."
if [ -f "auto_eval.sh" ]; then
    echo "   ✅ auto_eval.sh 存在"
else
    echo "   ❌ auto_eval.sh 不存在"
    exit 1
fi

if [ -f "auto_eval.py" ]; then
    echo "   ✅ auto_eval.py 存在"
else
    echo "   ❌ auto_eval.py 不存在"
    exit 1
fi

# 2. 检查可执行权限
echo ""
echo "2️⃣  检查可执行权限..."
if [ -x "auto_eval.sh" ]; then
    echo "   ✅ auto_eval.sh 可执行"
else
    echo "   ⚠️  auto_eval.sh 没有可执行权限，正在修复..."
    chmod +x auto_eval.sh
fi

if [ -x "auto_eval.py" ]; then
    echo "   ✅ auto_eval.py 可执行"
else
    echo "   ⚠️  auto_eval.py 没有可执行权限，正在修复..."
    chmod +x auto_eval.py
fi

# 3. 检查Python
echo ""
echo "3️⃣  检查Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✅ Python已安装: $PYTHON_VERSION"
else
    echo "   ❌ Python3未安装"
    exit 1
fi

# 4. 检查Conda (可选)
echo ""
echo "4️⃣  检查Conda (可选)..."
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    echo "   ✅ Conda已安装: $CONDA_VERSION"
else
    echo "   ℹ️  Conda未安装 (可选功能)"
fi

# 5. 检查Git
echo ""
echo "5️⃣  检查Git..."
if command -v git &> /dev/null; then
    GIT_VERSION=$(git --version)
    echo "   ✅ Git已安装: $GIT_VERSION"
else
    echo "   ❌ Git未安装 (必需)"
    exit 1
fi

# 6. 运行环境检查
echo ""
echo "6️⃣  检查Python依赖..."
if [ -f "check_env.py" ]; then
    python3 check_env.py
    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================="
        echo "✅ 所有检查通过！"
        echo "=================================="
        echo ""
        echo "你可以开始使用自动评估脚本："
        echo ""
        echo "方式1 (Bash):"
        echo "  1. 编辑 auto_eval.sh 修改配置"
        echo "  2. ./auto_eval.sh"
        echo ""
        echo "方式2 (Python):"
        echo "  python3 auto_eval.py --model_url <模型地址> --dataset algebra --conda_env eval_env"
        echo ""
    else
        echo ""
        echo "=================================="
        echo "⚠️  依赖检查失败"
        echo "=================================="
        echo "请先安装缺失的依赖包"
        exit 1
    fi
else
    echo "   ⚠️  check_env.py 不存在，跳过依赖检查"
fi
