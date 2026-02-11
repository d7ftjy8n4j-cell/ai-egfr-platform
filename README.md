# 🧬 EGFR抑制剂智能预测系统

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![RDKit](https://img.shields.io/badge/RDKit-3D9970?logo=python&logoColor=white)](https://www.rdkit.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

基于**双引擎**（随机森林 + 图神经网络）的表皮生长因子受体（EGFR）抑制剂活性预测平台，支持3D分子可视化、药效团分析和蛋白配体相互作用预测。

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ 核心功能

### 🔬 双引擎预测
- **随机森林模型 (RF)**：基于200+分子描述符的传统机器学习模型
  - AUC: 0.855 | Accuracy: 0.830
- **图神经网络模型 (GNN)**：基于GCN架构的深度学习模型
  - AUC: 0.808 | Accuracy: 0.765
- **集成预测**：双模型交叉验证，提供置信度评估

### 🎨 3D分子可视化
- 支持SMILES结构的3D渲染
- 多种显示样式：stick、sphere、cartoon
- 可交互式旋转、缩放

### 🔍 高级分析
- **药效团分析**：识别药效团特征和毒性预警
- **分子性质计算**：分子量、脂溶性、氢键供体/受体等
- **蛋白配体对接**：模拟EGFR蛋白与配体的相互作用

---

## 🚀 快速开始

### 本地运行

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd ai-egfr-platform

# 2. 创建虚拟环境
conda create -n egfr python=3.10
conda activate egfr

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用
streamlit run app.py
```

### Streamlit Cloud 部署

本项目已配置为可直接部署到 [Streamlit Cloud](https://share.streamlit.io)：

1. 推送代码到 GitHub
2. 登录 [share.streamlit.io](https://share.streamlit.io)
3. 选择此仓库并部署

---

## 📁 项目结构

```
.
├── 🐍 app.py                      # 主应用入口
├── 📦 requirements.txt            # Python依赖
├── 📦 packages.txt                # 系统依赖（Streamlit Cloud）
├── 🐳 Dockerfile.dockerfile       # Docker配置
│
├── 🧠 模型文件
│   ├── rf_egfr_model_final.pkl       # 随机森林模型 (~3GB)
│   ├── gcn_egfr_best_model.pth       # GNN模型 (~179MB)
│   └── feature_names.json            # 特征名称映射
│
├── 📊 可视化资源
│   ├── feature_importance.png        # 特征重要性图
│   └── gcn_confusion_matrix.png      # GNN混淆矩阵
│
└── 🔧 功能模块
    ├── real_predictor.py            # RF预测引擎
    ├── gnn_predictor.py             # GNN预测引擎
    ├── fallback_predictor.py        # 备用预测器
    ├── chem_filter.py               # 化学过滤器
    ├── chem_insight_safe.py         # 化学洞察分析
    ├── molecule_utils.py            # 分子工具集
    ├── structure_viz.py             # 3D结构可视化
    ├── pharmacophore_streamlit.py   # 药效团分析
    └── protein_ligand_streamlit.py  # 蛋白配体分析
```

---

## 🎯 使用指南

### 输入 SMILES
在输入框中输入分子的 SMILES 字符串，例如：
```
COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1
```
（吉非替尼/Gefitinib，EGFR抑制剂）

### 获取预测结果
1. 系统会自动验证 SMILES 格式
2. RF 和 GNN 模型分别进行预测
3. 显示综合预测结果和置信度

### 高级功能
- **3D 视图**：查看分子的三维结构
- **药效团分析**：分析药效团特征和毒性警报
- **分子性质**：详细理化性质报告

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **Web框架** | Streamlit 1.29.0 |
| **机器学习** | scikit-learn 1.3.2, PyTorch 2.1.2 |
| **图神经网络** | PyTorch Geometric 2.4.0 |
| **化学信息学** | RDKit 2022.9.5 |
| **3D可视化** | py3Dmol, stmol |
| **数据处理** | pandas, numpy |

---

## 📊 模型性能

| 模型 | AUC | 准确率 | 特征维度 |
|------|-----|--------|----------|
| 随机森林 (RF) | 0.855 | 83.0% | 200+ 分子描述符 |
| 图神经网络 (GNN) | 0.808 | 76.5% | 12维节点特征 |

---

## 📝 依赖说明

### 主要依赖版本
```
streamlit==1.29.0
scikit-learn==1.3.2
torch==2.1.2+cpu
torch-geometric==2.4.0
rdkit-pypi==2022.9.5
py3Dmol==2.0.4
```

### 完整依赖
详见 [`requirements.txt`](requirements.txt)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

## 🙏 致谢

- 模型训练基于公开EGFR生物活性数据集
- 3D可视化基于 [py3Dmol](https://3Dmol.csb.pitt.edu/)
- 化学信息学工具由 [RDKit](https://www.rdkit.org) 提供

---

<div align="center">

**Made with ❤️ for Drug Discovery**

</div>
