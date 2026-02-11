# Streamlit Cloud 依赖问题解决方案

## 问题现状

在 Streamlit Cloud 上部署时遇到依赖冲突：
- `rich` 版本 14.3.2 与 streamlit 1.29.0 不兼容
- `stmol` 仍然残留（应移除）
- `plip` 未安装
- 点击"查看分子结构"后崩溃

## 🎯 解决方法

### 方法一：使用 `setup.sh` 自动脚本（推荐）

创建一个 `setup.sh` 脚本，Streamlit Cloud 会在构建时自动执行：

```bash
# 新建文件：.streamlit/setup.sh
# 或直接使用项目根目录的 setup.sh

#!/bin/bash
# Streamlit Cloud 环境修复脚本

set -e

echo "🔧 开始修复 Streamlit Cloud 环境..."

# 升级 pip
python -m pip install --upgrade pip --quiet

# 强制卸载冲突包
pip uninstall -y streamlit stmol rich markdown-it-py pygments ipywidgets 2>/dev/null || true

# 安装严格锁定的版本
pip install "rich==13.7.1" --quiet
pip install "markdown-it-py==2.2.0" --quiet
pip install "pygments==2.17.2" --quiet
pip install "ipywidgets==7.6.3" --quiet

# 安装 streamlit
pip install "streamlit==1.29.0" --quiet

# 安装其他依赖
pip install -r requirements.txt --quiet || true

echo "✅ 环境修复完成"
```

### 方法二：修改 `.streamlit/config.toml`（最简单）

在 `.streamlit/config.toml` 中添加依赖锁定：

```toml
[server]
maxUploadSize = 200
```

然后在项目根目录创建 `packages.txt`（如果还没有）：

```
libxrender1
libsm6
libxext6
libfontconfig1
libgl1-mesa-glx
libxcomposite1
libxcursor1
libxdamage1
libxfixes3
libxi6
libxrandr2
libxrender1
libxss1
libxtst6
```

### 方法三：使用 `.prebuild` 脚本（推荐用于复杂修复）

创建 `.streamlit/prebuild.py`：

```python
#!/usr/bin/env python
"""
Streamlit Cloud 预构建脚本
在依赖安装之前执行
"""

import subprocess
import sys

def run_command(cmd):
    """执行命令并打印输出"""
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        # 不终止，继续执行
    return result.returncode

if __name__ == "__main__":
    print("🔧 开始 Streamlit Cloud 环境修复...")
    
    # 升级 pip
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # 卸载冲突包
    run_command([sys.executable, "-m", "pip", "uninstall", "-y",
                "streamlit", "stmol", "rich", "markdown-it-py",
                "pygments", "ipywidgets"])
    
    # 按正确顺序安装
    run_command([sys.executable, "-m", "pip", "install", "rich==13.7.1"])
    run_command([sys.executable, "-m", "pip", "install", "markdown-it-py==2.2.0"])
    run_command([sys.executable, "-m", "pip", "install", "pygments==2.17.2"])
    run_command([sys.executable, "-m", "pip", "install", "ipywidgets==7.6.3"])
    run_command([sys.executable, "-m", "pip", "install", "streamlit==1.29.0"])
    
    print("✅ 预构建完成")
```

## 📝 实际操作步骤

### 步骤 1：创建修复脚本

在项目根目录创建 `.streamlit/setup.sh`：

```bash
#!/bin/bash
python -m pip install --upgrade pip --quiet
pip uninstall -y streamlit stmol rich markdown-it-py pygments ipywidgets 2>/dev/null || true
pip install "rich==13.7.1" "markdown-it-py==2.2.0" "pygments==2.17.2" "ipywidgets==7.6.3" --quiet
pip install "streamlit==1.29.0" --quiet
```

### 步骤 2：提交到 GitHub

```bash
git add .streamlit/setup.sh
git commit -m "添加 Streamlit Cloud 环境修复脚本"
git push
```

### 步骤 3：在 Streamlit Cloud 中配置

1. 登录 [share.streamlit.io](https://share.streamlit.io)
2. 进入你的应用设置
3. 在 **Advanced settings** → **Pre-build commands** 中添加：
   ```bash
   bash .streamlit/setup.sh
   ```

### 步骤 4：重新部署

点击 Streamlit Cloud 的 **Rerun** 按钮，或推送代码触发重新部署。

---

## 🔍 验证修复

部署完成后，在应用中：

1. 打开侧边栏的 **"🔍 环境诊断"**
2. 查看所有包的版本状态
3. 确认没有依赖冲突

---

## ⚠️ 如果仍然失败

### 终极方案：禁用有问题的功能

如果 `plip` 或 `py3Dmol` 持续安装失败：

1. 在 `requirements.txt` 中完全注释掉它们
2. 应用会自动进入降级模式
3. 核心预测功能不受影响

---

## 📋 快速命令参考

```bash
# 本地测试修复脚本
bash .streamlit/setup.sh

# 本地运行验证
streamlit run app.py

# 本地环境诊断
python diagnose_env.py

# 提交到 GitHub
git add -A
git commit -m "修复 Streamlit Cloud 依赖问题"
git push
```

---

## 💡 重要提示

1. **Streamlit Cloud 每次重新部署都会重新安装依赖**
2. **所以修复脚本会在每次部署时自动执行**
3. **如果遇到持续失败，考虑简化依赖，只保留核心功能**

---

## ✅ 预期结果

修复后，你的应用应该：
- ✅ 能够正常启动
- ✅ GNN 预测功能正常
- ✅ 3D 分子可视化可用（如果 py3Dmol 安装成功）
- ⚠️ PLIP 分析功能可能受限（降级模式，不影响核心功能）
