# 🚀 《药尘光》项目部署指南

## 📋 问题诊断总结

您的项目在Streamlit Cloud部署时遇到了三个核心问题：

### 问题1️⃣: 模型与环境不兼容（最致命）
```
ModuleNotFoundError: No module named 'numpy._core'
```
**原因**: RF模型在numpy 1.26+上保存，云端使用numpy 1.24.4，内部模块路径不同

### 问题2️⃣: PyTorch Geometric依赖安装失败
```
Getting requirements to build wheel ... ModuleNotFoundError: No module named 'torch'
```
**原因**: torch-scatter等扩展包需要从源码编译，云端环境检测不到已安装的torch

### 问题3️⃣: Streamlit依赖版本冲突
```
streamlit 1.29.0 requires rich<14,>=10.14.0, but you have rich 14.3.2
```
**原因**: rich包被意外升级，与streamlit要求冲突

---

## ✅ 已完成的修复

### 1. 修复 requirements.txt
- ✅ 使用预编译的torch-geometric包，避免从源码编译
- ✅ 锁定rich版本，避免版本冲突

### 2. app.py RF模型已禁用
- ✅ 第136行: `RF_PREDICTOR_AVAILABLE = False`
- ✅ 首次部署只使用GNN模型

### 3. 创建的辅助工具
- ✅ `rebuild_model.py` - 重建兼容的RF模型
- ✅ `disable_rf_model.py` - 禁用RF模型
- ✅ `check_deployment.py` - 部署前环境检查

---

## 🎯 部署步骤（推荐顺序）

### 阶段1️⃣: 首次部署 - GNN单模型验证

#### 步骤1: 检查当前配置
```bash
cd C:\Users\dadamingli\Desktop\ai-egfr-platform
```

确认以下文件已正确更新：
- ✅ `requirements.txt` 已修改
- ✅ `app.py` 中 `RF_PREDICTOR_AVAILABLE = False`

#### 步骤2: 提交到GitHub
```bash
git add requirements.txt app.py
git commit -m "修复部署问题：使用预编译torch-geometric包，禁用RF模型"
git push
```

#### 步骤3: 部署到Streamlit Cloud
1. 登录 [share.streamlit.io](https://share.streamlit.io)
2. 点击 "New app"
3. 选择你的GitHub仓库
4. 选择分支：通常是 `main` 或 `master`
5. Main file path: `app.py`
6. 点击 "Deploy"

#### 步骤4: 监控部署日志
- 查看部署进度和日志
- 如果成功，访问应用URL测试功能
- 使用以下SMILES测试：
  ```
  CN1CCN(CC1)C2=NC3=C(N2)NC(NC3=O)C
  ```

---

### 阶段2️⃣: 成功后启用RF模型

#### 步骤1: 重建兼容的RF模型
在您的开发环境（aidd_competition）中运行：
```bash
cd C:\Users\dadamingli\Desktop\ai-egfr-platform
conda activate aidd_competition
python rebuild_model.py
```

这会创建 `rf_egfr_model_compatible.pkl`，与numpy 1.24.4兼容。

#### 步骤2: 更新app.py使用兼容模型
在 `real_predictor.py` 中，将模型路径从：
```python
model = joblib.load('rf_egfr_model_final.pkl')
```
改为：
```python
model = joblib.load('rf_egfr_model_compatible.pkl')
```

#### 步骤3: 启用RF模型
在 `app.py` 第136行，将：
```python
RF_PREDICTOR_AVAILABLE = False
```
改为：
```python
RF_PREDICTOR_AVAILABLE = True
```

#### 步骤4: 重新部署
```bash
git add app.py real_predictor.py rf_egfr_model_compatible.pkl
git commit -m "启用兼容的RF模型"
git push
```

---

### 阶段3️⃣: 集成高级功能（可选）

待基础部署成功后，可以逐步集成：
- 药效团特征
- 3D分子可视化
- 更多模型对比分析

---

## 🔧 快速命令参考

### 检查部署环境
```bash
python check_deployment.py
```

### 禁用RF模型
```bash
python disable_rf_model.py
```

### 重建兼容模型
```bash
python rebuild_model.py
```

### 本地测试应用
```bash
streamlit run app.py
```

---

## 📊 预期结果

### 成功部署的标志：
1. ✅ Streamlit Cloud部署日志无错误
2. ✅ 应用可以正常访问
3. ✅ 输入SMILES后能看到GNN预测结果
4. ✅ 侧边栏显示"✅ GNN预测器就绪"

### 首次部署预期状态：
- GNN模型: ✅ 就绪
- RF模型: ⏸️ 离线（正常，暂时禁用）

### 双模型部署成功后：
- GNN模型: ✅ 就绪
- RF模型: ✅ 就绪

---

## ⚠️ 常见问题排查

### Q1: 部署时torch-geometric安装失败
**原因**: 可能是Python版本不兼容
**解决**: 确认Streamlit Cloud使用Python 3.9-3.11

### Q2: GNN模型加载失败
**原因**: 模型文件缺失或路径错误
**解决**: 确认 `gcn_egfr_best_model.pth` 已提交到仓库

### Q3: 应用启动后立即崩溃
**原因**: 代码逻辑错误或依赖缺失
**解决**: 查看Streamlit Cloud的完整日志

### Q4: RF模型启用后加载失败
**原因**: numpy版本不兼容
**解决**: 必须先运行 `rebuild_model.py` 创建兼容版本

---

## 💡 最佳实践

1. **渐进式部署**: 不要试图一次性部署所有功能
2. **小步验证**: 每次只改变一个变量，验证后再继续
3. **查看日志**: 遇到问题首先查看完整的部署日志
4. **本地测试**: 每次修改后先本地测试 `streamlit run app.py`

---

## 📝 当前文件清单

### 核心文件
- ✅ `app.py` - 主应用（RF模型已禁用）
- ✅ `requirements.txt` - 依赖配置（已修复）
- ✅ `gnn_predictor.py` - GNN预测器
- ✅ `real_predictor.py` - RF预测器
- ✅ `gcn_egfr_best_model.pth` - GNN模型

### 模型文件
- ⚠️ `rf_egfr_model_final.pkl` - 原始RF模型（numpy 1.26+）
- ⚠️ `rf_egfr_model_compatible.pkl` - 兼容RF模型（需创建）

### 辅助工具
- ✅ `rebuild_model.py` - 重建兼容模型
- ✅ `disable_rf_model.py` - 禁用RF模型
- ✅ `check_deployment.py` - 环境检查

---

## 🎉 部署成功！

一旦部署成功，您将拥有一个功能完整的EGFR抑制剂智能预测系统！

**项目亮点**:
- 🧬 双引擎预测（RF + GNN）
- 📊 可视化分析
- 🎨 专业的UI界面
- ⚡ 实时预测

**访问地址**: [Streamlit Cloud部署后的URL]

---

祝部署顺利！如有问题，随时查看日志并参考本指南。
