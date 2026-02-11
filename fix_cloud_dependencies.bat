@echo off
REM ========== 强制清洁安装脚本 - 解决云端依赖冲突 (Windows版本) ==========
REM 此脚本会彻底清理冲突包并严格按requirements.txt安装

echo =========================================
echo 鍋 开始强制清洁安装...
echo =========================================

REM 1. 升级pip
echo 鍭 步骤1: 升级pip...
python -m pip install --upgrade pip --quiet

REM 2. 强制卸载所有可能冲突的包
echo 步骤2: 清理冲突包...
pip uninstall -y streamlit stmol rich markdown-it-py pygments ipywidgets plip py3Dmol altair >nul 2>&1
echo 旧包清理完成

REM 3. 安装严格锁定的streamlit依赖（防止自动升级）
echo 鍭 步骤3: 安装锁定的Streamlit核心依赖...
pip install "rich==13.7.1" --quiet
pip install "markdown-it-py==2.2.0" --quiet
pip install "pygments==2.17.2" --quiet
pip install "ipywidgets==7.6.3" --quiet
echo Streamlit核心依赖安装完成

REM 4. 安装Streamlit
echo 鍭 步骤4: 安装Streamlit...
pip install "streamlit==1.29.0" --quiet
echo Streamlit 1.29.0 安装完成

REM 5. 安装requirements.txt中的其他依赖（跳过可能有问题的包）
echo 鍭 步骤5: 安装requirements.txt中的依赖（跳过PLIP）...
pip install -r requirements.txt --quiet || echo 警告: 部分包安装失败

REM 尝试单独安装plip（允许失败）
echo 鍭 尝试安装PLIP（可选，允许失败）...
pip install "plip>=2.2.0" || echo 警告: PLIP安装失败，应用将进入降级模式

echo 其他依赖安装完成

REM 6. 验证安装结果
echo.
echo =========================================
echo 鍭 验证安装结果...
echo =========================================
pip list | findstr /i "streamlit rich ipywidgets stmol py3Dmol plip rdkit"

echo.
echo =========================================
echo 安装完成！
echo =========================================
echo.
echo 如果plip安装失败，应用会自动进入降级模式（不影响核心功能）
echo 现在可以运行: streamlit run app.py
pause
