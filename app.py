"""
app.py - EGFR抑制剂智能预测系统（双引擎版）
集成：真实随机森林模型 + 真实GNN模型
版本：1.0 - 双模型集成版 20.205.243.166
"""

# ========== 基础导入与设置 ==========
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# 设置页面
st.set_page_config(
    page_title="EGFR抑制剂智能预测系统 (双引擎)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 1. 双模型预测器导入 ==========
RF_PREDICTOR_AVAILABLE = False
GNN_PREDICTOR_AVAILABLE = False

# 导入随机森林预测器
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from real_predictor import RealEGFRPredictor
    RF_PREDICTOR_AVAILABLE = True
    st.sidebar.success("✅ 随机森林预测器就绪")
except ImportError as e:
    st.sidebar.warning(f"⚠️ 随机森林预测器导入失败: {str(e)[:50]}...")

# 导入GNN预测器
try:
    # 检查并修复GNN预测器文件路径问题
    gnn_predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnn_predictor.py")
    if os.path.exists(gnn_predictor_path):
        from gnn_predictor import GCNPredictor
        GNN_PREDICTOR_AVAILABLE = True
        st.sidebar.success("✅ GNN预测器就绪")
    else:
        st.sidebar.warning("⚠️ gnn_predictor.py文件未找到")
except ImportError as e:
    error_msg = str(e)
    st.sidebar.warning(f"⚠️ GNN预测器导入失败: {error_msg[:80]}...")

# ========== 2. 应用标题与介绍 ==========
st.title("🧬 EGFR抑制剂智能预测系统")
st.markdown("""
**双引擎预测系统** - 集成传统机器学习与深度学习技术  
- **🧪 标准模式**: 基于随机森林与分子描述符  
- **🧠 高级模式**: 基于图神经网络(GNN)与分子结构图  
- **📊 对比分析**: 双模型结果对比与一致性验证
""")

# 系统状态指示器
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("随机森林模型", "就绪" if RF_PREDICTOR_AVAILABLE else "离线", 
             "AUC: 0.855" if RF_PREDICTOR_AVAILABLE else "N/A")
with col2:
    st.metric("GNN模型", "就绪" if GNN_PREDICTOR_AVAILABLE else "离线", 
             "AUC: 0.808" if GNN_PREDICTOR_AVAILABLE else "N/A")
with col3:
    st.metric("数据集", "5,568化合物", "58.5%活性")

# ========== 3. 初始化预测器 ==========
@st.cache_resource
def init_predictors():
    """初始化预测器（缓存以提高性能）"""
    predictors = {}
    
    # 初始化随机森林预测器
    if RF_PREDICTOR_AVAILABLE:
        try:
            # 使用绝对路径确保能找到模型文件
            desktop_path = os.path.expanduser("~/Desktop")
            rf_model_path = os.path.join(desktop_path, "rf_egfr_model_final.pkl")
            
            if os.path.exists(rf_model_path):
                predictors['rf'] = RealEGFRPredictor()
                st.sidebar.info(f"✓ RF模型: {os.path.basename(rf_model_path)}")
            else:
                st.sidebar.error(f"❌ RF模型文件未找到: {rf_model_path}")
        except Exception as e:
            st.sidebar.error(f"❌ RF预测器初始化失败: {str(e)[:50]}")
    
    # 初始化GNN预测器
    if GNN_PREDICTOR_AVAILABLE:
        try:
            # 使用绝对路径
            desktop_path = os.path.expanduser("~/Desktop")
            gnn_model_path = os.path.join(desktop_path, "gcn_egfr_best_model.pth")
            
            if os.path.exists(gnn_model_path):
                predictors['gnn'] = GCNPredictor(model_path=gnn_model_path, device='cpu')
                st.sidebar.info(f"✓ GNN模型: {os.path.basename(gnn_model_path)}")
            else:
                st.sidebar.error(f"❌ GNN模型文件未找到: {gnn_model_path}")
        except Exception as e:
            st.sidebar.error(f"❌ GNN预测器初始化失败: {str(e)[:50]}")
    
    return predictors

# 初始化所有预测器
predictors = init_predictors()

# ========== 4. 辅助函数 ==========
def display_rf_result(result, model_name="随机森林"):
    """显示随机森林预测结果"""
    if "error" in result:
        st.error(f"❌ {model_name}预测失败: {result['error']}")
        return
    
    # 结果卡片
    if result['prediction'] == 1:
        st.success(f"## ✅ {model_name}: 活性化合物")
    else:
        st.error(f"## ❌ {model_name}: 非活性化合物")
    
    # 指标显示
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("活性概率", f"{result['probability_active']:.3f}")
    with col_b:
        st.metric("置信度", result.get('confidence', '中'))
    with col_c:
        st.metric("AUC参考", "0.855")
    
    # 特征解释（如果可用）
    if result.get('explanation'):
        with st.expander(f"📊 {model_name}决策依据"):
            for i, (feat, imp) in enumerate(zip(result['explanation']['top_features'], 
                                               result['explanation']['top_importance']), 1):
                st.write(f"**{i}. {feat}** - 重要性: `{imp:.4f}`")

def display_gnn_result(result, model_name="GNN图神经网络"):
    """显示GNN预测结果"""
    if not result["success"]:
        st.error(f"❌ {model_name}预测失败: {result.get('error', '未知错误')}")
        return
    
    # 结果卡片
    if result['prediction'] == 1:
        st.success(f"## ✅ {model_name}: 活性化合物")
    else:
        st.error(f"## ❌ {model_name}: 非活性化合物")
    
    # 指标显示
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("活性概率", f"{result['probability_active']:.4f}")
    with col_b:
        st.metric("置信度", result.get('confidence', '中'))
    with col_c:
        st.metric("AUC参考", "0.808")
    
    # 模型信息
    with st.expander(f"🧠 {model_name}详情"):
        st.write(f"**模型类型**: {result.get('model_type', 'GCN图卷积网络')}")
        st.write(f"**测试集准确率**: {result.get('model_accuracy', 0.7652):.3f}")
        st.write(f"**测试集AUC**: {result.get('model_auc', 0.8081):.3f}")
        st.write("**原理**: 将分子视为图结构（原子为节点，化学键为边），使用图卷积网络直接学习分子结构特征")

def compare_results(rf_result, gnn_result):
    """比较两个模型的预测结果"""
    st.markdown("---")
    st.subheader("📊 双模型对比分析")
    
    # 创建对比表格
    comparison_data = []
    
    if "error" not in rf_result:
        comparison_data.append({
            "模型": "随机森林 (RF)",
            "预测": "活性" if rf_result['prediction'] == 1 else "非活性",
            "活性概率": f"{rf_result['probability_active']:.4f}",
            "置信度": rf_result.get('confidence', '中'),
            "AUC": "0.855",
            "原理": "基于200+个RDKit分子描述符"
        })
    
    if gnn_result.get('success', False):
        comparison_data.append({
            "模型": "图神经网络 (GNN)",
            "预测": "活性" if gnn_result['prediction'] == 1 else "非活性",
            "活性概率": f"{gnn_result['probability_active']:.4f}",
            "置信度": gnn_result.get('confidence', '中'),
            "AUC": "0.808",
            "原理": "基于分子图结构直接学习"
        })
    
    if comparison_data:
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)
        
        # 结论分析
        if len(comparison_data) == 2:
            rf_pred = comparison_data[0]['预测']
            gnn_pred = comparison_data[1]['预测']
            rf_prob = float(comparison_data[0]['活性概率'])
            gnn_prob = float(comparison_data[1]['活性概率'])
            
            if rf_pred == gnn_pred:
                st.success("✅ **双模型结论一致**，结果可靠性高")
                if abs(rf_prob - gnn_prob) < 0.2:
                    st.info("两个模型的预测概率接近，进一步验证了结果的可信度")
            else:
                st.warning("⚠️ **双模型结论不一致**")
                st.markdown("""
                **可能原因分析**:
                1. **分子结构特殊**: GNN对图拓扑结构敏感，RF依赖于预设描述符
                2. **模型视角不同**: GNN是"端到端"学习，RF是"特征工程+学习"
                3. **建议**: 可结合分子相似性搜索进一步验证
                """)

# ========== 5. 主界面 - 标签页设计 ==========
tab1, tab2, tab3, tab4 = st.tabs(["🧪 分子预测", "📊 模型分析", "🔬 技术详情", "📚 关于项目"])

with tab1:
    st.header("🧪 分子活性预测")
    
    # 预测模式选择
    prediction_mode = st.radio(
        "**选择预测模式**",
        ["🤖 标准模式 (随机森林)", "🧠 高级模式 (GNN图神经网络)", "⚡ 双模型对比", "📚 示例分子"],
        horizontal=True,
        key="pred_mode"
    )
    
    # 输入区域
    if prediction_mode != "📚 示例分子":
        smiles_input = st.text_area(
            "**输入SMILES字符串**",
            value="Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",
            height=100,
            help="输入分子SMILES表示，如: Cc1cc(C)c(/C=C2\\C(=O)Nc3ncnc(Nc4ccc(F)c(Cl)c4)c32)oc1C",
            key="smiles_input"
        )
    
    # 示例分子选择
    if prediction_mode == "📚 示例分子":
        example_molecules = {
            "吉非替尼 (EGFR抑制剂)": "COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "高活性EGFR抑制剂": "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",
            "阿司匹林 (非活性对照)": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "咖啡因 (非活性对照)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        }
        
        selected_example = st.selectbox("选择示例分子:", list(example_molecules.keys()))
        smiles_input = example_molecules[selected_example]
        st.code(smiles_input)
        
        # 使用选择的示例进行预测
        prediction_mode = "⚡ 双模型对比"
    
    # 预测按钮
    if prediction_mode != "📚 示例分子" and st.button("🚀 开始预测", type="primary", use_container_width=True):
        if smiles_input.strip():
            with st.spinner("正在分析分子..."):
                
                # ========== 标准模式 - 随机森林 ==========
                if prediction_mode.startswith("🤖 标准模式"):
                    if 'rf' in predictors:
                        rf_result = predictors['rf'].predict(smiles_input.strip())
                        display_rf_result(rf_result)
                    else:
                        st.error("随机森林预测器不可用")
                
                # ========== 高级模式 - GNN ==========
                elif prediction_mode.startswith("🧠 高级模式"):
                    if 'gnn' in predictors:
                        gnn_result = predictors['gnn'].predict(smiles_input.strip())
                        display_gnn_result(gnn_result)
                        
                        # 显示分子结构
                        try:
                            from rdkit import Chem
                            from rdkit.Chem import Draw
                            from PIL import Image
                            import io
                            
                            mol = Chem.MolFromSmiles(smiles_input.strip())
                            if mol:
                                img = Draw.MolToImage(mol, size=(300, 200))
                                st.image(img, caption="分子2D结构 (由GNN解析)")
                        except:
                            pass
                    else:
                        st.error("GNN预测器不可用")
                
                # ========== 双模型对比模式 ==========
                elif prediction_mode.startswith("⚡ 双模型对比"):
                    col_left, col_right = st.columns(2)
                    
                    # 左侧：随机森林结果
                    with col_left:
                        if 'rf' in predictors:
                            rf_result = predictors['rf'].predict(smiles_input.strip())
                            display_rf_result(rf_result, "随机森林模型")
                        else:
                            st.warning("随机森林模型不可用")
                    
                    # 右侧：GNN结果
                    with col_right:
                        if 'gnn' in predictors:
                            gnn_result = predictors['gnn'].predict(smiles_input.strip())
                            display_gnn_result(gnn_result, "GNN模型")
                        else:
                            st.warning("GNN模型不可用")
                    
                    # 对比分析
                    if 'rf' in predictors and 'gnn' in predictors:
                        compare_results(rf_result, gnn_result)
        else:
            st.warning("请输入有效的SMILES字符串")

with tab2:
    st.header("📊 模型性能分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("随机森林模型")
        st.metric("AUC", "0.855", "优秀")
        st.metric("准确率", "0.830", "良好")
        st.metric("特征数量", "200+", "RDKit描述符")
        
        with st.expander("📈 特征重要性"):
            st.image("feature_importance.png" if os.path.exists("feature_importance.png") else 
                    "https://via.placeholder.com/400x200?text=特征重要性图", 
                    caption="随机森林特征重要性排序")
    
    with col2:
        st.subheader("GNN模型")
        st.metric("AUC", "0.808", "良好")
        st.metric("准确率", "0.765", "良好")
        st.metric("节点特征", "12维", "原子级特征")
        
        with st.expander("📈 训练历史"):
            st.image("gcn_training_history.png" if os.path.exists("gcn_training_history.png") else 
                    "https://via.placeholder.com/400x200?text=GNN训练曲线", 
                    caption="GNN训练损失与准确率曲线")
    
    # 模型对比说明
    st.markdown("---")
    st.subheader("🎯 模型选择建议")
    
    advice_data = {
        "推荐场景": ["已知分子描述符", "分子结构图", "需要解释性", "追求前沿技术"],
        "随机森林": ["✅ 优秀", "❌ 不适用", "✅ 特征重要性", "较传统"],
        "GNN": ["❌ 不需要", "✅ 优秀", "❌ 黑盒性", "✅ 前沿"]
    }
    
    st.table(pd.DataFrame(advice_data))

with tab3:
    st.header("🔬 技术实现详情")
    
    st.markdown("""
    ### 🏗️ 系统架构
    
    **双引擎预测架构**:
    ```
    输入层 (SMILES)
        ├── 随机森林分支 → RDKit特征提取 → 随机森林模型 → 预测结果
        └── GNN分支 → 分子图转换 → 图卷积网络 → 预测结果
    ```
    
    ### 🔧 技术栈
    
    | 组件 | 技术选择 | 用途 |
    |------|----------|------|
    | **前端界面** | Streamlit | 交互式Web应用 |
    | **传统ML** | Scikit-learn + RDKit | 随机森林模型训练与预测 |
    | **深度学习** | PyTorch + PyTorch Geometric | GNN模型训练与预测 |
    | **化学计算** | RDKit | 分子特征计算与可视化 |
    | **数据管理** | Pandas + NumPy | 数据处理与分析 |
    
    ### 📐 特征工程对比
    
    **随机森林特征** (200+维度):
    - 物理化学性质: LogP, 分子量, 氢键供体/受体等
    - 结构特征: 芳香环数, 可旋转键数, 拓扑极性表面积等
    - 原子计数: C, N, O, F等原子类型统计
    
    **GNN特征** (12维原子特征):
    - 原子级特征: 原子序数, 杂化类型, 形式电荷, 芳香性等
    - 键级特征: 键类型, 共轭性, 环内键等
    - 通过图卷积层自动学习分子结构表示
    
    ### 🎯 模型性能
    
    | 指标 | 随机森林 | GNN | 说明 |
    |------|----------|-----|------|
    | **AUC** | 0.855 | 0.808 | 随机森林略优 |
    | **准确率** | 0.830 | 0.765 | 随机森林更稳定 |
    | **可解释性** | 高 | 中 | RF有特征重要性 |
    | **泛化能力** | 强 | 较强 | 均表现良好 |
    | **创新性** | 传统 | 前沿 | GNN代表AI趋势 |
    """)

with tab4:
    st.header("📚 关于项目")
    
    st.markdown("""
    ### 🎯 项目简介
    
    **EGFR抑制剂双引擎智能预测系统**是一个集成了传统机器学习与深度学习的计算药学平台。
    本项目展示了如何将不同范式的AI技术应用于药物发现中的关键问题——EGFR抑制剂活性预测。
    
    ### 🏆 项目特色
    
    1. **双模型架构**: 同时实现随机森林与图神经网络，提供多角度预测
    2. **对比分析**: 自动对比不同模型的预测结果，提高可靠性
    3. **完整流程**: 涵盖从数据获取、特征工程、模型训练到应用部署的全流程
    4. **真实数据**: 基于5,568个真实EGFR化合物的ChEMBL数据
    5. **可解释性**: 提供特征重要性分析，增强结果可信度
    
    ### 🔬 科学价值
    
    - **方法学对比**: 系统比较了"特征工程+传统ML"与"端到端深度学习"在药物发现中的应用
    - **技术集成**: 展示了如何将RDKit、Scikit-learn、PyTorch等工具整合到完整工作流中
    - **可复现性**: 所有代码开源，数据可公开获取，保证研究的可复现性
    
    ### 📁 项目文件
    
    项目包含以下核心文件:
    
    - `app.py` - 主应用程序 (当前文件)
    - `real_predictor.py` - 随机森林预测器
    - `gnn_predictor.py` - GNN图神经网络预测器
    - `rf_egfr_model_final.pkl` - 随机森林模型
    - `gcn_egfr_best_model.pth` - GNN模型
    - `egfr_compounds_clean.csv` - 清洗后的数据集
    - `feature_names.json` - 特征名称列表
    
    ### 👨‍🔬 致谢
    
    本项目基于以下开源教育资源构建:
    
    - **TeachOpenCADD** 平台提供的T001、T007、T035等教程
    - **ChEMBL** 数据库提供的EGFR抑制剂活性数据
    - **RDKit** 开源化学信息学工具包
    - **PyTorch Geometric** 图神经网络库
    
    ### 📄 许可证
    
    本项目仅供学习和研究使用。如需用于商业目的，请联系开发者获取授权。
    """)
    
    # 添加时间戳
    st.caption(f"系统生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ========== 6. 侧边栏信息 ==========
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 模型状态
    st.subheader("模型状态")
    
    rf_status = "✅ 在线" if 'rf' in predictors else "❌ 离线"
    gnn_status = "✅ 在线" if 'gnn' in predictors else "❌ 离线"
    
    st.write(f"- 随机森林: {rf_status}")
    st.write(f"- GNN模型: {gnn_status}")
    
    # 使用统计
    st.subheader("📈 使用统计")
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    
    st.metric("总预测次数", st.session_state.prediction_count)
    
    # 快速链接
    st.subheader("🔗 快速操作")
    
    if st.button("🔄 重置所有预测"):
        st.session_state.prediction_count = 0
        st.rerun()
    
    if st.button("📥 导出当前结果"):
        st.info("导出功能开发中...")
    
    # 系统信息
    st.subheader("ℹ️ 系统信息")
    st.write(f"Python: {sys.version.split()[0]}")
    st.write("Streamlit: 1.28.0")
    st.write(f"工作目录: {os.getcwd()}")

# ========== 7. 页脚 ==========
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🧬 EGFR抑制剂双引擎智能预测系统 | 传统ML + 深度学习集成 | © 2026
    <br>
    <small>基于TeachOpenCADD教程构建 | 仅供学术研究使用</small>
    </div>
    """,
    unsafe_allow_html=True
)

# ========== 8. 自动更新预测计数 ==========
if 'last_smiles' not in st.session_state:
    st.session_state.last_smiles = ""

# 检查是否有新的预测
if st.session_state.get('smiles_input', '') != st.session_state.last_smiles:
    if st.session_state.get('smiles_input', '').strip():
        st.session_state.prediction_count += 1
        st.session_state.last_smiles = st.session_state.get('smiles_input', '')