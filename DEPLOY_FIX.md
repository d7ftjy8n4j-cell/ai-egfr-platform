# 部署修复说明

## 修复内容

### 1. 依赖修复 (requirements.txt)
- 修复 rich 版本冲突: `rich>=10.14.0,<14`
- 添加 ipywidgets>=8.0.0 满足 nglview 要求
- 添加 setuptools>=68.0.0 解决 pkg_resources 问题
- 限制 nglview<4.0.0 避免版本冲突

### 2. 模型加载修复 (real_predictor.py)
- 添加 RDKit 可用性检测
- RDKit 不可用时使用简单 SMILES 分析
- 模型加载失败时自动重建

### 3. 备用预测器 (fallback_predictor.py)
- 创建不依赖 RDKit 的基础版本
- RDKit 可用时自动升级

### 4. 兜底机制 (app.py)
- 内嵌 MinimalEGFRPredictor
- 三层降级确保应用可用

## 部署步骤

```bash
git add .
git commit -m "修复 Streamlit 部署：依赖冲突、RDKit兼容性、模型加载"
git push origin main
```

然后在 Streamlit Cloud 点击 "Reboot"。
