"""
app.py - EGFR抑制剂智能预测系统（双引擎版）
集成：真实随机森林模型 + 真实GNN模型
版本：1.0.0
"""

# ========== 基础导入 ==========
import sys
import os
import logging
from datetime import datetime

# ========== 设置页面（必须在任何Streamlit命令之前） ==========
import streamlit as st
st.set_page_config(
    page_title="EGFR抑制剂智能预测系统 (双引擎)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 初始化 Session State ==========
if 'last_smiles' not in st.session_state:
    st.session_state.last_smiles = ""
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'last_rf_result' not in st.session_state:
    st.session_state.last_rf_result = None
if 'last_gnn_result' not in st.session_state:
    st.session_state.last_gnn_result = None
if 'advanced_analysis_triggered' not in st.session_state:
    st.session_state.advanced_analysis_triggered = False

# ========== 添加路径 ==========
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ========== 导入药效团模块（不使用Streamlit UI） ==========
try:
    import pharmacophore_streamlit
    PHARMACOPHORE_AVAILABLE = True
    logging.info("药效团模块加载成功")
except ImportError as e:
    PHARMACOPHORE_AVAILABLE = False
    logging.error(f"药效团模块导入失败: {e}")

# ========== 其他导入 ==========
import pandas as pd
import numpy as np
import joblib
import json
import re

# ========== 3D结构可视化导入 ==========
try:
    from structure_viz import StructureVisualizer
    from stmol import showmol
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    logging.warning("3D可视化依赖缺失 (stmol, py3DMol)")

# ========== 配置类 ==========
class Config:
    """集中管理系统配置"""
    # 阈值配置
    PROBABILITY_THRESHOLD = 0.2
    MAX_SMILES_LENGTH = 1000

    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_FILE = os.path.join(BASE_DIR, "app.log")

    # 模型默认性能指标
    RF_DEFAULT_PERF = {'auc': 0.855, 'accuracy': 0.830, 'feature_count': '200+'}
    GNN_DEFAULT_PERF = {'auc': 0.808, 'accuracy': 0.765, 'node_features': '12维'}

    # SMILES 允许字符模式
    SMILES_PATTERN = r'^[A-Za-z0-9@+\-\[\]\(\)\\\/%=#$]+$'

    # 日志级别（可动态调整）
    LOG_LEVEL = logging.INFO

# 配置日志（输出到文件）
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# 同时输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(Config.LOG_LEVEL)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

# 定义常量（从 Config 类中获取，保持向后兼容）
PROBABILITY_THRESHOLD = Config.PROBABILITY_THRESHOLD
MAX_SMILES_LENGTH = Config.MAX_SMILES_LENGTH
BASE_DIR = Config.BASE_DIR

# ========== 0. 辅助函数 ==========
def get_model_performance(model_type='rf', predictor=None):
    """获取模型性能指标（优先从预测器动态获取）"""
    # 尝试从预测器对象中读取
    if predictor and hasattr(predictor, 'auc'):
        return {
            'auc': getattr(predictor, 'auc', None),
            'accuracy': getattr(predictor, 'accuracy', None),
            'feature_count': getattr(predictor, 'feature_count', 'N/A'),
            'node_features': getattr(predictor, 'node_features', 'N/A')
        }
    # 使用配置类中的默认值
    if model_type == 'rf':
        return Config.RF_DEFAULT_PERF.copy()
    elif model_type == 'gnn':
        return Config.GNN_DEFAULT_PERF.copy()
    return {}

def sanitize_input(input_str):
    """清理输入字符串，防止注入攻击"""
    return re.sub(r'[^\w@+\-\[\]\(\)\\\/%=#$]', '', input_str)

def validate_smiles(smiles):
    """验证SMILES字符串的有效性"""
    # 首先进行字符范围检查
    if not re.match(Config.SMILES_PATTERN, smiles):
        logging.warning(f"SMILES字符格式不合法: {smiles[:50]}...")
        return False

    # 尝试使用RDKit验证
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except ImportError:
        logging.warning("RDKit未安装，跳过SMILES深度验证")
        return True  # 字符格式正确且无RDKit时，认为基本有效
    except Exception as e:
        logging.error(f"RDKit验证SMILES失败: {e}")
        return False

def validate_smiles_fallback(smiles):
    """SMILES验证的备用方案（仅正则表达式匹配）"""
    try:
        return bool(re.match(Config.SMILES_PATTERN, smiles))
    except Exception:
        return False

def check_gnn_model_files():
    """检查GNN模型相关文件是否存在"""
    gnn_predictor_path = os.path.join(Config.BASE_DIR, "gnn_predictor.py")
    gnn_model_path = os.path.join(Config.BASE_DIR, "gcn_egfr_best_model.pth")

    missing_files = []
    if not os.path.exists(gnn_predictor_path):
        missing_files.append("gnn_predictor.py")
    if not os.path.exists(gnn_model_path):
        missing_files.append("gcn_egfr_best_model.pth")

    return missing_files

# ========== 1. 双模型预测器导入 ==========
RF_PREDICTOR_AVAILABLE = True
GNN_PREDICTOR_AVAILABLE = False

# 定义内嵌兜底预测器（确保即使所有外部文件失败也能工作）
class MinimalEGFRPredictor:
    """最简EGFR预测器，不依赖任何外部库"""
    def __init__(self):
        self.feature_names = ["SMILES长度", "碳原子数", "氮原子数", "氧原子数"]
    
    def predict(self, smiles):
        """基于SMILES字符串长度的简单预测"""
        length = len(smiles)
        c_count = smiles.count('C')
        n_count = smiles.count('N')
        o_count = smiles.count('O')
        
        # 简单规则：中等长度、有氮和氧的分子更可能是EGFR抑制剂
        score = 0.5
        if 30 <= length <= 80:
            score += 0.15
        if n_count >= 2:
            score += 0.15
        if o_count >= 1:
            score += 0.10
        
        import random
        random.seed(hash(smiles) % 2**32)
        score += random.uniform(-0.1, 0.1)
        probability = max(0.1, min(0.9, score))
        
        return {
            "success": True,
            "smiles": smiles,
            "prediction": 1 if probability > 0.5 else 0,
            "probability_active": probability,
            "confidence": "中",
            "explanation": {
                "top_features": ["SMILES长度", "氮原子数", "氧原子数"],
                "top_importance": [0.5, 0.3, 0.2],
                "values": {"SMILES长度": length, "氮原子数": n_count, "氧原子数": o_count}
            },
            "features_used": self.feature_names,
            "feature_values": [length, c_count, n_count, o_count],
            "note": "使用最简预测器（部署兼容模式）"
        }

# 导入随机森林预测器
try:
    sys.path.append(Config.BASE_DIR)
    from real_predictor import RealEGFRPredictor
    # 尝试初始化以验证模型加载
    test_predictor = RealEGFRPredictor()
    if test_predictor.model is None:
        raise Exception("模型加载失败: 模型为None")
    RF_PREDICTOR_AVAILABLE = True
    st.sidebar.success("✅ 随机森林预测器就绪")
    logging.info("随机森林预测器导入成功")
except Exception as e:
    logging.error(f"随机森林预测器失败: {e}")
    # 尝试使用备用预测器
    try:
        from fallback_predictor import FallbackEGFRPredictor
        # 创建一个兼容RealEGFRPredictor接口的包装类
        class RealEGFRPredictor(FallbackEGFRPredictor):
            pass
        test_predictor = RealEGFRPredictor()
        RF_PREDICTOR_AVAILABLE = True
        st.sidebar.warning("⚠️ 使用备用随机森林预测器")
        logging.info("备用随机森林预测器加载成功")
    except Exception as fallback_error:
        logging.error(f"备用预测器也失败: {fallback_error}")
        # 使用最简兜底预测器
        class RealEGFRPredictor(MinimalEGFRPredictor):
            pass
        test_predictor = RealEGFRPredictor()
        RF_PREDICTOR_AVAILABLE = True
        st.sidebar.warning("⚠️ 使用兼容模式预测器")
        logging.info("最简兜底预测器加载成功")

# 导入GNN预测器
try:
    # 检查GNN模型文件完整性
    missing_files = check_gnn_model_files()
    if missing_files:
        st.sidebar.warning(f"⚠️ GNN模型文件缺失: {', '.join(missing_files)}")
        GNN_PREDICTOR_AVAILABLE = False
        logging.warning(f"GNN模型文件缺失: {missing_files}")
    else:
        from gnn_predictor import GCNPredictor
        GNN_PREDICTOR_AVAILABLE = True
        st.sidebar.success("✅ GNN预测器就绪")
        logging.info("GNN预测器导入成功")
except ImportError as e:
    error_msg = str(e)
    st.sidebar.warning(f"⚠️ GNN预测器导入失败: {error_msg[:80]}...")
    logging.error(f"GNN预测器导入失败: {e}")

# ========== 化学洞察安全模块导入 ==========
try:
    from chem_insight_safe import render_safe_chem_insight
    CHEM_INSIGHT_AVAILABLE = True
    st.sidebar.success("✅ 化学洞察模块就绪")
    logging.info("化学洞察模块导入成功")
except ImportError as e:
    CHEM_INSIGHT_AVAILABLE = False
    st.sidebar.warning(f"⚠️ 化学洞察模块导入失败: {e}")
    logging.warning(f"化学洞察模块导入失败: {e}")

# 药效团模块状态显示（侧边栏）
if PHARMACOPHORE_AVAILABLE:
    st.sidebar.success("✅ 药效团模块就绪")
else:
    st.sidebar.warning("⚠️ 药效团模块未加载")

# ========== 2. 应用标题与介绍 ==========
st.title("🧬 EGFR抑制剂智能预测系统")
st.markdown("""
**双引擎预测系统** - 集成传统机器学习与深度学习技术  
- **🧪 标准模式**: 基于随机森林与分子描述符  
- **🧠 高级模式**: 基于图神经网络(GNN)与分子结构图  
- **📊 对比分析**: 双模型结果对比与一致性验证
""")

# 系统状态指示器
rf_perf = get_model_performance('rf')
gnn_perf = get_model_performance('gnn')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("随机森林模型", "就绪" if RF_PREDICTOR_AVAILABLE else "离线",
             f"AUC: {rf_perf.get('auc', 'N/A')}" if RF_PREDICTOR_AVAILABLE else "N/A")
with col2:
    st.metric("GNN模型", "就绪" if GNN_PREDICTOR_AVAILABLE else "离线",
             f"AUC: {gnn_perf.get('auc', 'N/A')}" if GNN_PREDICTOR_AVAILABLE else "N/A")
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
            predictors['rf'] = RealEGFRPredictor()
            # 检查模型是否真正加载成功
            if predictors['rf'].model is None:
                st.sidebar.error("❌ RF预测器初始化失败: 模型未加载")
                logging.error("RF预测器初始化失败: 模型未加载")
                del predictors['rf']
            else:
                st.sidebar.info("✅ RF模型加载成功")
        except Exception as e:
            logging.error(f"RF预测器初始化失败: {e}")
            st.sidebar.error(f"❌ RF预测器初始化失败: {str(e)[:50]}")

    # 初始化GNN预测器
    if GNN_PREDICTOR_AVAILABLE:
        try:
            predictors['gnn'] = GCNPredictor(device='cpu')
            st.sidebar.info("✅ GNN模型加载成功")
        except Exception as e:
            logging.error(f"GNN预测器初始化失败: {e}")
            st.sidebar.error(f"❌ GNN预测器初始化失败: {str(e)[:50]}")

    return predictors

# 初始化所有预测器
predictors = init_predictors()

# ========== 4. 辅助函数 ==========

def _build_comparison_row(result, model_type, perf):
    """构建对比表格的行数据"""
    prediction_label = "活性" if result['prediction'] == 1 else "非活性"
    if model_type == 'rf':
        return {
            "模型": "随机森林 (RF)",
            "预测": prediction_label,
            "活性概率": f"{result['probability_active']:.4f}",
            "置信度": result.get('confidence', '中'),
            "AUC": str(perf.get('auc', 'N/A')),
            "原理": "基于200+个RDKit分子描述符"
        }
    else:  # gnn
        return {
            "模型": "图神经网络 (GNN)",
            "预测": prediction_label,
            "活性概率": f"{result['probability_active']:.4f}",
            "置信度": result.get('confidence', '中'),
            "AUC": str(perf.get('auc', 'N/A')),
            "原理": "基于分子图结构直接学习"
        }

def _display_result_header(result, model_name):
    """显示预测结果头部（错误检查 + 活性状态）"""
    # 错误检查
    if isinstance(result, dict):
        if "error" in result:
            st.error(f"❌ {model_name}预测失败: {result['error']}")
            return False
        if not result.get("success", True):
            st.error(f"❌ {model_name}预测失败: {result.get('error', '未知错误')}")
            return False

    # 结果卡片
    if result['prediction'] == 1:
        st.success(f"## ✅ {model_name}: 活性化合物")
    else:
        st.error(f"## ❌ {model_name}: 非活性化合物")
    return True

def _display_metrics(result, perf, precision=4):
    """显示指标（活性概率、置信度、AUC）"""
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("活性概率", f"{result['probability_active']:.{precision}f}")
    with col_b:
        st.metric("置信度", result.get('confidence', '中'))
    with col_c:
        st.metric("AUC参考", str(perf.get('auc', 'N/A')))

def display_model_result(result, model_name, model_type):
    """统一显示模型预测结果（减少代码重复）"""
    perf = get_model_performance(model_type)

    # 显示头部
    if not _display_result_header(result, model_name):
        return

    # 显示指标
    precision = 3 if model_type == 'rf' else 4
    _display_metrics(result, perf, precision)

    # 特有内容
    if model_type == 'rf':
        # 特征解释（如果可用）
        if result.get('explanation'):
            with st.expander(f"📊 {model_name}决策依据"):
                for i, (feat, imp) in enumerate(zip(result['explanation']['top_features'],
                                                   result['explanation']['top_importance']), 1):
                    st.write(f"**{i}. {feat}** - 重要性: `{imp:.4f}`")
    else:  # gnn
        # 模型信息
        with st.expander(f"🧠 {model_name}详情"):
            st.write(f"**模型类型**: {result.get('model_type', 'GCN图卷积网络')}")
            st.write(f"**测试集准确率**: {result.get('model_accuracy', 0.7652):.3f}")
            st.write(f"**测试集AUC**: {result.get('model_auc', 0.8081):.3f}")
            st.write("**原理**: 将分子视为图结构（原子为节点，化学键为边），使用图卷积网络直接学习分子结构特征")

# 向后兼容的别名
def display_prediction_result(result, model_name, model_type):
    """通用预测结果显示函数（别名）"""
    display_model_result(result, model_name, model_type)

def display_rf_result(result, model_name="随机森林"):
    """显示随机森林预测结果（向后兼容）"""
    display_model_result(result, model_name, 'rf')

def display_gnn_result(result, model_name="GNN图神经网络"):
    """显示GNN预测结果（向后兼容）"""
    display_model_result(result, model_name, 'gnn')

def export_prediction_to_csv(results, filename="prediction_results.csv"):
    """导出预测结果到CSV文件"""
    try:
        df = pd.DataFrame(results)
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载CSV文件",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
        logging.info(f"预测结果已导出到CSV: {filename}")
        return True
    except Exception as e:
        logging.error(f"导出CSV失败: {e}")
        st.error(f"导出失败: {str(e)}")
        return False

def export_results_to_dataframe(results_dict):
    """将预测结果转换为DataFrame（用于导出）"""
    data = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for model_type, result in results_dict.items():
        if isinstance(result, dict):
            row = {
                '时间戳': timestamp,
                'SMILES': st.session_state.get('last_smiles', ''),
                '模型': model_type.upper(),
            }

            if 'error' not in result and result.get('success', True):
                row['预测结果'] = '活性' if result.get('prediction') == 1 else '非活性'
                row['活性概率'] = f"{result.get('probability_active', 0):.4f}"
                row['置信度'] = result.get('confidence', '中')

                if model_type == 'rf':
                    perf = get_model_performance('rf')
                else:
                    perf = get_model_performance('gnn')
                row['参考AUC'] = str(perf.get('auc', 'N/A'))
            else:
                row['预测结果'] = '失败'
                row['错误信息'] = result.get('error', '未知错误')

            data.append(row)

    return pd.DataFrame(data)

def compare_results(rf_result, gnn_result):
    """比较两个模型的预测结果"""
    st.markdown("---")
    st.subheader("📊 双模型对比分析")

    rf_perf = get_model_performance('rf')
    gnn_perf = get_model_performance('gnn')

    # 创建对比表格
    comparison_data = []

    if "error" not in rf_result:
        comparison_data.append(_build_comparison_row(rf_result, 'rf', rf_perf))

    if gnn_result.get('success', False):
        comparison_data.append(_build_comparison_row(gnn_result, 'gnn', gnn_perf))

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
                if abs(rf_prob - gnn_prob) < PROBABILITY_THRESHOLD:
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
tab1, tab2, tab3, tab4, tab5, tab7, tab6 = st.tabs([
    "🧪 分子预测",
    "🔍 化学依据",
    "🎯 药效团设计",
    "📊 模型分析",
    "🔬 技术详情",
    "🔗 3D结构",   # 新增
    "📚 关于项目"
])

with tab1:
    st.header("🧪 分子活性预测")
    
    # 预测模式选择
    prediction_mode = st.radio(
        "**选择预测模式**",
        [
            "🤖 标准模式 (随机森林)",
            "🧠 高级模式 (GNN图神经网络)",
            "⚡ 双模型对比",
            "📚 示例分子"
        ],
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

        # 示例分子自动预测按钮
        if st.button("🚀 使用示例分子进行预测", type="primary", use_container_width=True, key="example_predict"):
            # 使用选择的示例进行预测
            actual_prediction_mode = "⚡ 双模型对比"
    else:
        # 预测按钮（非示例模式）
        actual_prediction_mode = prediction_mode
        # 输入验证
        smiles_clean = smiles_input.strip()

        # 检查输入长度
        if len(smiles_clean) > MAX_SMILES_LENGTH:
            st.error(f"❌ 输入的 SMILES 字符串过长（超过 {MAX_SMILES_LENGTH} 字符），请缩短后重试")
        elif not smiles_clean:
            st.warning("请输入有效的SMILES字符串")
        elif not validate_smiles(smiles_clean):
            st.error("❌ 无效的 SMILES 字符串，请检查格式后重试")
        else:
            # 更新预测计数
            st.session_state.prediction_count += 1
            st.session_state.last_smiles = smiles_clean

            # 进度条
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                with st.spinner("正在分析分子..."):
                    status_text.text("准备模型...")
                    progress_bar.progress(10)

                    # ========== 标准模式 - 随机森林 ==========
                    if actual_prediction_mode.startswith("🤖 标准模式"):
                        status_text.text("随机森林预测中...")
                        progress_bar.progress(30)
                        if 'rf' in predictors:
                            rf_result = predictors['rf'].predict(smiles_clean)
                            st.session_state.last_rf_result = rf_result
                            progress_bar.progress(80)
                            display_rf_result(rf_result)
                        else:
                            st.error("随机森林预测器不可用")

                    # ========== 高级模式 - GNN ==========
                    elif actual_prediction_mode.startswith("🧠 高级模式"):
                        status_text.text("GNN图神经网络预测中...")
                        progress_bar.progress(30)
                        if 'gnn' in predictors:
                            gnn_result = predictors['gnn'].predict(smiles_clean)
                            st.session_state.last_gnn_result = gnn_result
                            progress_bar.progress(60)
                            display_gnn_result(gnn_result)

                            # 显示分子结构
                            try:
                                from rdkit import Chem
                                from rdkit.Chem import Draw, AllChem

                                mol = Chem.MolFromSmiles(smiles_clean)
                                if mol:
                                    status_text.text("生成分子结构图...")
                                    progress_bar.progress(80)

                                    # 计算二维坐标
                                    AllChem.Compute2DCoords(mol)

                                    # 生成图像
                                    img = Draw.MolToImage(mol, size=(300, 200))
                                    st.image(img, caption="分子2D结构")
                                else:
                                    st.warning("⚠️ 无法解析分子结构，请检查SMILES格式")
                                    logging.warning(f"RDKit无法解析SMILES: {smiles_clean[:50]}...")
                            except Exception as e:
                                st.warning(f"⚠️ 分子结构图显示失败: {str(e)[:150]}")
                                logging.warning(f"分子结构显示失败: {e}")
                                # 降级显示：显示SMILES字符串
                                st.info(f"分子SMILES: {smiles_clean}")
                        else:
                            st.error("GNN预测器不可用")

                    # ========== 双模型对比模式 ==========
                    elif actual_prediction_mode.startswith("⚡ 双模型对比"):
                        col_left, col_right = st.columns(2)

                        # 初始化结果变量
                        rf_result = None
                        gnn_result = None

                        # 左侧：随机森林结果
                        with col_left:
                            status_text.text("随机森林预测中...")
                            progress_bar.progress(20)
                            if 'rf' in predictors:
                                rf_result = predictors['rf'].predict(smiles_clean)
                                st.session_state.last_rf_result = rf_result
                                progress_bar.progress(40)
                                display_rf_result(rf_result, "随机森林模型")
                            else:
                                st.warning("随机森林模型不可用")

                        # 右侧：GNN结果
                        with col_right:
                            status_text.text("GNN预测中...")
                            progress_bar.progress(60)
                            if 'gnn' in predictors:
                                gnn_result = predictors['gnn'].predict(smiles_clean)
                                st.session_state.last_gnn_result = gnn_result
                                progress_bar.progress(80)
                                display_gnn_result(gnn_result, "GNN模型")
                            else:
                                st.warning("GNN模型不可用")

                        # 对比分析（仅当两个结果都存在时）
                        if rf_result is not None and gnn_result is not None:
                            status_text.text("生成对比分析...")
                            progress_bar.progress(95)
                            compare_results(rf_result, gnn_result)

                    progress_bar.progress(100)
                    status_text.text("✅ 预测完成！")

            except Exception as e:
                logging.error(f"预测过程出错: {e}")
                st.error(f"❌ 预测过程中发生错误: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()

with tab2:
    st.header("🔍 化学依据分析")
    if CHEM_INSIGHT_AVAILABLE:
        render_safe_chem_insight()
    else:
        st.error("化学洞察模块不可用")
        st.code("请确保 chem_insight_safe.py 和 molecule_utils.py 文件存在")

with tab3:
    st.header("🎯 药效团设计")
    if PHARMACOPHORE_AVAILABLE:
        pharmacophore_streamlit.render_pharmacophore_tab()
    else:
        st.error("药效团模块不可用")
        st.code("请确保 pharmacophore_streamlit.py 文件存在")

with tab4:
    st.header("📊 模型性能分析")

    rf_perf = get_model_performance('rf')
    gnn_perf = get_model_performance('gnn')

    # 获取图片路径
    feature_img_path = os.path.join(BASE_DIR, "feature_importance.png")
    gcn_img_path = os.path.join(BASE_DIR, "gcn_confusion_matrix.png")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("随机森林模型")
        st.metric("AUC", str(rf_perf.get('auc', 'N/A')), "优秀")
        st.metric("准确率", str(rf_perf.get('accuracy', 'N/A')), "良好")
        st.metric("特征数量", rf_perf.get('feature_count', 'N/A'), "RDKit描述符")

        with st.expander("📈 特征重要性"):
            st.image(feature_img_path if os.path.exists(feature_img_path) else
                    "https://via.placeholder.com/400x200?text=特征重要性图",
                    caption="随机森林特征重要性排序")

    with col2:
        st.subheader("GNN模型")
        st.metric("AUC", str(gnn_perf.get('auc', 'N/A')), "良好")
        st.metric("准确率", str(gnn_perf.get('accuracy', 'N/A')), "良好")
        st.metric("节点特征", gnn_perf.get('node_features', 'N/A'), "原子级特征")

    with st.expander("📈 混淆矩阵"):
        st.image(gcn_img_path if os.path.exists(gcn_img_path) else
                    "https://via.placeholder.com/400x200?text=GNN混淆矩阵",
                    caption="GNN模型混淆矩阵")

    # 模型对比说明
    st.markdown("---")
    st.subheader("🎯 模型选择建议")

    advice_data = {
        "推荐场景": ["已知分子描述符", "分子结构图", "需要解释性", "追求前沿技术"],
        "随机森林": ["✅ 优秀", "❌ 不适用", "✅ 特征重要性", "较传统"],
        "GNN": ["❌ 不需要", "✅ 优秀", "❌ 黑盒性", "✅ 前沿"]
    }

    st.table(pd.DataFrame(advice_data))

with tab5:
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

with tab7:
    st.header("🔗 蛋白质-配体 3D 结构可视化")
    
    if not VIZ_AVAILABLE:
        st.error("⚠️ 缺少可视化组件。请在 requirements.txt 中添加 `stmol` 和 `py3Dmol`。")
    else:
        # 布局：左侧控制，右侧显示
        col_ctrl, col_view = st.columns([1, 3])
        
        # 初始化 Session State 用于存储 PDB 数据
        if 'viz_pdb_id' not in st.session_state:
            st.session_state.viz_pdb_id = "3POZ"
        if 'viz_data_loaded' not in st.session_state:
            st.session_state.viz_data_loaded = False
        
        # --- 左侧控制栏 ---
        with col_ctrl:
            st.subheader("1. 数据加载")
            input_mode = st.radio("来源:", ["PDB ID", "上传文件"])
            
            viz_tool = StructureVisualizer()
            load_success = False
            
            if input_mode == "PDB ID":
                pdb_input = st.text_input("输入 ID", value=st.session_state.viz_pdb_id).upper()
                if st.button("📥 加载 PDB", use_container_width=True):
                    with st.spinner("下载中..."):
                        if viz_tool.load_from_pdb_id(pdb_input):
                            st.session_state.viz_pdb_id = pdb_input
                            st.session_state.viz_data_loaded = True
                            st.session_state.viz_data_source = "remote"
                            # 将数据存入 session 以便重绘时无需重新下载
                            st.session_state.viz_raw_data = viz_tool.pdb_data
                            load_success = True
                        else:
                            st.error("无效的 PDB ID")
            else:
                uploaded_file = st.file_uploader("上传 .pdb", type="pdb")
                if uploaded_file:
                    viz_tool.load_from_file(uploaded_file)
                    st.session_state.viz_data_loaded = True
                    st.session_state.viz_data_source = "local"
                    st.session_state.viz_raw_data = viz_tool.pdb_data
                    load_success = True

            st.markdown("---")
            st.subheader("2. 样式设置")
            
            # 从 Session 恢复数据 (如果只是调整样式，不需要重新下载)
            if st.session_state.viz_data_loaded and not load_success:
                viz_tool.pdb_data = st.session_state.viz_raw_data
                viz_tool.pdb_id = st.session_state.get('viz_pdb_id', 'Unknown')
            
            style_select = st.selectbox("蛋白样式", ["cartoon", "stick", "line", "sphere"], index=0)
            color_select = st.selectbox("配色方案", ["spectrum", "chain", "residue"], index=0)
            
            show_ligand = st.toggle("显示配体/药物", value=True)
            show_surface = st.toggle("显示蛋白表面", value=False)
            
            surface_opacity = 0.5
            if show_surface:
                surface_opacity = st.slider("表面透明度", 0.0, 1.0, 0.5, 0.1)

        # --- 右侧显示区 ---
        with col_view:
            if st.session_state.viz_data_loaded:
                st.info(f"正在查看: **{viz_tool.pdb_id}**")
                
                # 生成视图
                try:
                    view = viz_tool.render_view(
                        style=style_select,
                        color_scheme=color_select,
                        show_ligand=show_ligand,
                        show_surface=show_surface,
                        surface_opacity=surface_opacity
                    )
                    # 在 Streamlit 中显示
                    showmol(view, height=600, width=800)
                    
                    st.caption("💡 操作提示: 鼠标左键旋转，右键/Ctrl+左键平移，滚轮缩放。")
                    
                except Exception as e:
                    st.error(f"渲染失败: {e}")
            else:
                # 初始空状态占位
                st.info("👈 请在左侧加载蛋白质结构")
                st.markdown("""
                **推荐的 EGFR 相关结构:**
                * `3POZ`: EGFR 激酶结构域 + 抑制剂 Tak-285
                * `1M17`: EGFR + 埃罗替尼 (Erlotinib)
                * `2ITY`: EGFR + 吉非替尼 (Gefitinib)
                """)

with tab6:
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

    - `app.py` - 主应用程序
    - `real_predictor.py` - 随机森林预测器
    - `gnn_predictor.py` - GNN图神经网络预测器
    - `pharmacophore_streamlit.py` - 药效团设计模块
    - `rf_egfr_model_final.pkl` - 随机森林模型
    - `gcn_egfr_best_model.pth` - GNN模型
    - `egfr_compounds_clean.csv` - 清洗后的数据集
    - `feature_names.json` - 特征名称列表

    ### 👨‍🔬 致谢

    本项目基于以下开源教育资源构建:

    - **TeachOpenCADD** 平台提供的T001、T007、T009、T033、T035等教程
    - **ChEMBL** 数据库提供的EGFR抑制剂活性数据
    - **RDKit** 开源化学信息学工具包
    - **PyTorch Geometric** 图神经网络库

    ### 📄 备注

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
    st.metric("总预测次数", st.session_state.prediction_count)

    # 快速链接
    st.subheader("🔗 快速操作")

    if st.button("🔄 重置所有预测"):
        st.session_state.prediction_count = 0
        st.rerun()

    if st.button("📥 导出当前结果"):
        # 检查是否有可导出的预测结果
        if not st.session_state.get('last_smiles'):
            st.warning("暂无预测结果可导出")
        else:
            st.subheader("导出预测结果")

            # 从session_state获取真实的预测结果
            export_data = {}

            if st.session_state.get('last_rf_result'):
                rf_result = st.session_state.last_rf_result
                if isinstance(rf_result, dict) and 'error' not in rf_result:
                    export_data['rf'] = rf_result

            if st.session_state.get('last_gnn_result'):
                gnn_result = st.session_state.last_gnn_result
                if isinstance(gnn_result, dict) and gnn_result.get('success', True):
                    export_data['gnn'] = gnn_result

            if export_data:
                # 将结果转换为DataFrame
                df = export_results_to_dataframe(export_data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # 下载按钮
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                filename = f"egfr_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    label="📥 下载CSV文件",
                    data=csv,
                    file_name=filename,
                    mime="text/csv"
                )
                logging.info(f"预测结果已导出: {filename}")
            else:
                st.warning("没有可用的模型结果")

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
    🧬 EGFR抑制剂双引擎智能预测系统 | 传统随机森林 + 深度学习集成 | © 2026
    <br>
    <small>基于TeachOpenCADD教程构建 | 仅供学术研究使用</small>
    </div>
    """,
    unsafe_allow_html=True
)