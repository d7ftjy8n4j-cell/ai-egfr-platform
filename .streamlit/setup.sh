#!/bin/bash
# Streamlit Cloud 环境修复脚本
# 此脚本会在每次部署时自动执行

set -e

echo "========================================="
echo "🔧 开始修复 Streamlit Cloud 环境..."
echo "========================================="

# 升级 pip
echo "📦 步骤1: 升级 pip..."
python -m pip install --upgrade pip --quiet

# 强制卸载冲突包
echo "🗑️  步骤2: 卸载冲突包..."
pip uninstall -y streamlit stmol rich markdown-it-py pygments ipywidgets 2>/dev/null || true
echo "✓ 旧包清理完成"

# 安装严格锁定的版本（防止自动升级）
echo "📦 步骤3: 安装锁定的依赖..."
pip install "rich==13.7.1" --quiet
pip install "markdown-it-py==2.2.0" --quiet
pip install "pygments==2.17.2" --quiet
pip install "ipywidgets==7.6.3" --quiet
echo "✓ 核心依赖安装完成"

# 安装 streamlit
echo "📦 步骤4: 安装 Streamlit..."
pip install "streamlit==1.29.0" --quiet
echo "✓ Streamlit 安装完成"

# 尝试安装其他依赖
echo "📦 步骤5: 安装 requirements.txt..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt --quiet || echo "⚠️ 部分包安装失败，应用会自动降级"
    echo "✓ 依赖安装完成"
else
    echo "⚠️ requirements.txt 不存在，跳过"
fi

echo ""
echo "========================================="
echo "✅ 环境修复完成！"
echo "========================================="
