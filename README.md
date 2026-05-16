# 🧬 药尘光 · EGFR抑制剂智能发现与设计平台

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![RDKit](https://img.shields.io/badge/RDKit-3D9970?logo=python&logoColor=white)](https://www.rdkit.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

**药尘光** 是一款面向EGFR抑制剂的**教学友好型Web平台**，集成随机森林（RF）与图神经网络（GNN）双引擎，提供分子活性预测、药效团分析、3D可视化、ADME筛选等全流程功能。

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fd7ftjy8n4j-cell%2Fai-egfr-platform&label=Visitors&countColor=%23263759)

🔗 **在线体验**：[https://ai-egfr-platform.streamlit.app/](https://ai-egfr-platform.streamlit.app/)

---

## ✨ 核心功能

### 🔬 双引擎智能预测
- **随机森林 (RF)**：基于12个精选RDKit分子描述符（LogP、HBA、HBD、TPSA等）  
  ✅ 5折交叉验证 AUC = **0.867** | 准确率 = **0.782**
- **图神经网络 (GNN)**：3层GCN，13维原子特征，端到端学习分子拓扑  
  ✅ 5折交叉验证 AUC = **0.845** | 准确率 = **0.767**
- **集成决策**：双模型并行预测 + 一致性判断，提供置信度评估

### 🎨 3D分子可视化
- 支持蛋白-配体复合物（PDB ID/本地文件）交互式3D渲染
- 多种显示样式：cartoon、stick、sphere，可自定义配色
- 实时旋转、缩放、平移

### 🔍 高级分析
- **药效团分析**：从活性分子中提取氢键供/受体、疏水区、芳香环等特征，生成3D药效团模型
- **化学依据**：Lipinski五规则、PAINS/Brenk毒性警报、相似性搜索（Morgan指纹）
- **ADME筛选**：批量评估化合物的口服成药性与安全性

### 📊 模型解读
- 随机森林特征重要性排序
- GNN混淆矩阵与训练曲线
- 双模型结果对比表格

---

## 🚀 快速开始

### 本地运行（推荐使用conda）

```bash
# 1. 克隆仓库
git clone https://github.com/d7ftjy8n4j-cell/ai-egfr-platform.git
cd ai-egfr-platform

# 2. 创建虚拟环境
conda create -n egfr python=3.10
conda activate egfr

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用
streamlit run app.py
```

### Streamlit Cloud 一键部署

本项目已配置 `packages.txt` 和 `Dockerfile`，可直接部署到 [Streamlit Cloud](https://share.streamlit.io)：

1. 将代码推送到GitHub仓库
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 点击 "New app" → 选择仓库 → 设置主文件为 `app.py`
4. 点击 "Deploy"

---

## 📁 项目结构

```
.
├── app.py                          # 主应用（st.navigation 架构）
├── requirements.txt                # Python依赖
├── packages.txt                    # 系统依赖（供Streamlit Cloud）
├── Dockerfile.dockerfile           # Docker镜像配置
│
├── 🧠 模型文件
│   ├── rf_egfr_model_final.pkl     # 随机森林模型 (交叉验证训练)
│   ├── gcn_egfr_best_model.pth      # GNN模型 (交叉验证训练)
│   └── feature_names.json          # 12个特征名称列表
│
├── 📊 可视化资源
│   ├── feature_importance.png      # RF特征重要性图
│   ├── gcn_confusion_matrix.png    # GNN混淆矩阵
│   └── gcn_training_history.png    # GNN训练曲线
│
└── 🔧 功能模块
    ├── real_predictor.py           # 随机森林预测器
    ├── gnn_predictor.py            # GNN预测器
    ├── chem_filter.py              # ADME筛选器
    ├── chem_insight_safe.py         # 化学洞察模块
    ├── pharmacophore_streamlit.py  # 药效团分析
    ├── structure_viz.py            # 3D可视化引擎
    └── protein_ligand_streamlit.py # 蛋白-配体分析
```

---

## 🎯 使用指南

### 1. 输入SMILES
示例（吉非替尼）：
```
COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4
```

### 2. 选择预测模式
- 🤖 标准模式（随机森林）
- 🧠 高级模式（GNN）
- ⚡ 双模型对比

### 3. 查看结果
- 活性概率、置信度、AUC参考
- 随机森林特征重要性解释
- GNN模型结构说明

### 4. 扩展分析
- 切换至 **"药物筛选"** 标签页评估成药性
- 使用 **"药效团设计"** 生成3D药效团模型
- 进入 **"3D结构"** 加载PDB文件查看蛋白-配体相互作用

---

## 🛠️ 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| Web框架 | Streamlit | 1.29.0+ |
| 机器学习 | scikit-learn | 1.3.2 |
| 深度学习 | PyTorch / PyTorch Geometric | 2.1.2 / 2.4.0 |
| 化学信息学 | RDKit | 2022.9.5 |
| 3D可视化 | py3Dmol / stmol | 2.0.4 / 0.3.0 |
| 数据处理 | pandas / numpy | 1.5.3 / 1.24.4 |

---

## 📊 模型性能（5折交叉验证）

| 模型 | AUC | 准确率 | 关键特征/维度 |
|------|-----|--------|----------------|
| 随机森林 | **0.867 ± 0.005** | **0.782 ± 0.005** | MolLogP, NumHAcceptors, TPSA, 芳香环数等12个描述符 |
| GNN | **0.845 ± 0.008** | **0.767 ± 0.011** | 13维原子特征（原子序数、度、电荷、杂化等） |

> 数据来源：ChEMBL EGFR靶点（CHEMBL203），经IC50 (nM) 筛选、去重后共 **13,286** 个唯一化合物，活性阈值 pIC50 ≥ 7.0。

---

## 📝 依赖说明

核心依赖见 `requirements.txt`。注意：若在Streamlit Cloud部署，需同时包含 `packages.txt` 以安装系统级依赖（如 `autodock-vina`，用于结构亲和力评估模块）。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。本项目遵循开源精神，所有代码可供教学与科研复用。

---

## 📄 许可证

[MIT License](LICENSE)

---

## 🙏 致谢

- 数据：ChEMBL数据库（5,568 → 13,286条EGFR活性数据）
- 教程：TeachOpenCADD (T001, T007, T033, T035)
- 工具：RDKit, PyTorch Geometric, Streamlit, scikit-learn

---

<div align="center">

**双核驱动，理形相生**  
*从微观尘埃中寻找治愈之光*

[![GitHub stars](https://img.shields.io/github/stars/d7ftjy8n4j-cell/ai-egfr-platform?style=social)](https://github.com/d7ftjy8n4j-cell/ai-egfr-platform)

</div>
