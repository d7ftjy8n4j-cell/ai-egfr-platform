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
    page_title="药尘光 · EGFR抑制剂智能发现平台",
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
    VIZ_ERROR = None
except Exception as e:
    VIZ_AVAILABLE = False
    VIZ_ERROR = str(e)
    # 同时在后台打印详细错误
    import traceback
    logging.error(f"3D可视化模块导入失败: {e}")
    logging.error(traceback.format_exc())

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


# ========== 新增：缓存 3D 视图生成 ==========
# 使用 cache_resource 因为 view 是一个复杂的对象
@st.cache_resource
def get_3d_view(pdb_data, style, color_scheme, show_ligand, show_surface, surface_opacity):
    """
    缓存 3D 视图对象。
    只有当传入的参数发生变化时，才会重新创建 view 对象。
    这能有效防止页面无限刷新。
    """
    if not pdb_data:
        return None
    
    # 临时实例化 Visualizer 来利用它的逻辑
    # 注意：我们需要在这里引用 StructureVisualizer，确保它已导入
    from structure_viz import StructureVisualizer 
    
    viz_tool = StructureVisualizer()
    viz_tool.pdb_data = pdb_data
    
    # 调用渲染方法
    view = viz_tool.render_view(
        style=style,
        color_scheme=color_scheme,
        show_ligand=show_ligand,
        show_surface=show_surface,
        surface_opacity=surface_opacity
    )
    return view

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
    # st.sidebar.success("✅ 随机森林预测器就绪")
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
        # st.sidebar.success("✅ GNN预测器就绪")
        logging.info("GNN预测器导入成功")
except ImportError as e:
    error_msg = str(e)
    st.sidebar.warning(f"⚠️ GNN预测器导入失败: {error_msg[:80]}...")
    logging.error(f"GNN预测器导入失败: {e}")

# 化学洞察安全模块导入
try:
    from chem_insight_safe import render_safe_chem_insight
    CHEM_INSIGHT_AVAILABLE = True
    # st.sidebar.success("✅ 化学洞察模块就绪")
    logging.info("化学洞察模块导入成功")
except ImportError as e:
    CHEM_INSIGHT_AVAILABLE = False
    st.sidebar.warning(f"⚠️ 化学洞察模块导入失败: {e}")
    logging.warning(f"化学洞察模块导入失败: {e}")

# 药物筛选模块导入
try:
    from chem_filter import ADMEFilter, SubstructureFilter
    FILTER_AVAILABLE = True
    # st.sidebar.success("✅ 药物筛选模块就绪")
    logging.info("药物筛选模块导入成功")
except ImportError:
    FILTER_AVAILABLE = False
    st.sidebar.warning("⚠️ 药物筛选模块未加载")
    logging.warning("药物筛选模块导入失败")

# 药效团模块状态显示（侧边栏）
# if PHARMACOPHORE_AVAILABLE:
#     st.sidebar.success("✅ 药效团模块就绪")
# else:
#     st.sidebar.warning("⚠️ 药效团模块未加载")

# ========== 2. 应用标题与介绍 ==========
st.title("🧬 药尘光 · EGFR抑制剂智能发现平台")
st.markdown("""
**双引擎预测系统** —— 集成传统机器学习与深度学习技术
- **🧪 标准模式**: 基于随机森林与分子描述符
- **🧠 高级模式**: 基于图神经网络(GNN)与分子结构图
- **📊 对比分析**: 双模型结果对比与一致性验证
*"双核驱动，理形相生"*
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
                # st.sidebar.info("✅ RF模型加载成功")
                pass
        except Exception as e:
            logging.error(f"RF预测器初始化失败: {e}")
            st.sidebar.error(f"❌ RF预测器初始化失败: {str(e)[:50]}")

    # 初始化GNN预测器
    if GNN_PREDICTOR_AVAILABLE:
        try:
            predictors['gnn'] = GCNPredictor(device='cpu')
            # st.sidebar.info("✅ GNN模型加载成功")
            pass
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🧪 分子预测",        # 核心活性预测
    "🛡️ 药物筛选",        # 成药性与安全性
    "🔍 化学依据",        # 理化性质与相似性
    "🎯 药效团设计",      # 活性特征提取与设计指导
    "🔗 3D结构",          # 蛋白-配体三维可视化
    "📊 模型分析",        # 模型性能与特征重要性
    "🔬 技术详情",        # 技术实现细节
    "📚 关于项目"         # 背景与致谢
])

with tab1:
    st.header("🧪 分子活性预测")
    st.caption("输入 SMILES，选择预测模式，快速评估分子对 EGFR 的抑制活性。双模型对比可提高结果可靠性。")
    st.info(
        "🎓 **教学点**：对比随机森林（基于特征工程）与图神经网络（基于分子图结构）的预测结果，"
        "理解两种AI范式的差异。当两个模型结论不一致时，思考可能的原因（如分子中的特殊环结构）。"
    )

    # 预测模式选择
    prediction_mode = st.radio(
        "**选择预测模式**",
        [
            "🤖 标准模式 (随机森林)",
            "🧠 高级模式 (GNN图神经网络)",
            "⚡ 双模型对比"
        ],
        horizontal=True,
        key="pred_mode"
    )

    # 输入区域
    smiles_input = st.text_area(
        "**输入SMILES字符串**",
        value="Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",
        height=100,
        help="输入分子SMILES表示，如: Cc1cc(C)c(/C=C2\\C(=O)Nc3ncnc(Nc4ccc(F)c(Cl)c4)c32)oc1C",
        key="smiles_input"
    )

    # 预测按钮
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
    st.header("🛡️ 药物类属性与安全性筛选")
    st.caption("评估化合物的成药潜力：Lipinski 五规则（ADME）和毒性警报（PAINS/Brenk）。单分子或批量筛选。")
    st.info(
        "🎓 **教学点**：理解Lipinski五规则（分子量、LogP、氢键供体/受体）如何评估口服成药性，"
        "以及PAINS/Brenk子结构警报提示的潜在风险。"
    )

    if not FILTER_AVAILABLE:
        st.error("筛选模块未加载，请检查 chem_filter.py 文件")
    else:
        # 初始化筛选器
        adme_tool = ADMEFilter()
        struct_tool = SubstructureFilter()

        st.markdown("""
        本模块用于评估化合物的成药潜力，包括：
        1.  **ADME/Ro5**: Lipinski 五规则 (分子量、亲脂性、氢键供体/受体)
        2.  **毒性警报**: 筛查 PAINS (泛测定干扰化合物) 和 Brenk 不良子结构
        """)

        # 两个模式：单分子 vs 批量
        mode = st.radio("选择模式", ["单分子分析 (当前SMILES)", "批量数据集筛选"], horizontal=True)

        # --- 模式 1: 单分子分析 ---
        if mode == "单分子分析 (当前SMILES)":
            current_smiles = st.session_state.get('last_smiles', '')

            if not current_smiles:
                st.info("请先在「🧪 分子预测」页面输入并预测一个分子，或在下方手动输入。")
                current_smiles = st.text_input("输入 SMILES", value="CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCC")
            else:
                st.write(f"**当前分析分子**: `{current_smiles}`")

            if current_smiles and st.button("开始评估", type="primary"):
                col_res1, col_res2 = st.columns(2)

                # 1. Ro5 分析
                with col_res1:
                    st.subheader("1. Lipinski 五规则 (Ro5)")
                    ro5_res = adme_tool.calculate_ro5_properties(current_smiles)

                    if ro5_res['MW'] is not None:
                        # 使用 DataFrame 展示并高亮
                        res_df = pd.DataFrame(ro5_res).T
                        # 格式化
                        st.dataframe(res_df.style.format("{:.2f}", subset=["MW", "LogP"]), use_container_width=True)

                        if ro5_res['Pass_Ro5']:
                            st.success("✅ **通过 Ro5 筛选** (违反规则数 <= 1)")
                        else:
                            st.error("❌ **未通过 Ro5 筛选** (违反规则数 > 1)")

                        # 详细指标检查
                        st.caption("规则详情:")
                        st.write(f"- 分子量 {'✅' if ro5_res['MW']<=500 else '❌'} (≤500): {ro5_res['MW']:.1f}")
                        st.write(f"- LogP {'✅' if ro5_res['LogP']<=5 else '❌'} (≤5): {ro5_res['LogP']:.2f}")
                        st.write(f"- HBA {'✅' if ro5_res['HBA']<=10 else '❌'} (≤10): {ro5_res['HBA']}")
                        st.write(f"- HBD {'✅' if ro5_res['HBD']<=5 else '❌'} (≤5): {ro5_res['HBD']}")
                    else:
                        st.error("无法计算理化性质")

                # 2. 子结构分析
                with col_res2:
                    st.subheader("2. 不良子结构警报")
                    struct_res = struct_tool.check_single_molecule(current_smiles)

                    if "error" in struct_res:
                        st.error("SMILES 解析错误")
                    else:
                        # PAINS
                        if struct_res["PAINS_found"]:
                            st.error(f"⚠️ **发现 PAINS 警报**: {', '.join(struct_res['PAINS_names'])}")
                            st.warning("PAINS (Pan Assay Interference Compounds) 可能会导致实验假阳性。")
                        else:
                            st.success("✅ 未发现 PAINS 结构")

                        st.markdown("---")

                        # Brenk
                        if struct_res["Brenk_found"]:
                            st.warning(f"⚠️ **发现 Brenk 不良结构**: {', '.join(struct_res['Brenk_names'])}")
                            st.caption("这些结构可能具有毒性、代谢不稳定性或化学反应性。")
                        else:
                            st.success("✅ 未发现 Brenk 不良结构")

        # --- 模式 2: 批量筛选 ---
        else:
            uploaded_csv = st.file_uploader("上传分子列表 CSV (需包含 smiles 列)", type="csv")

            if uploaded_csv:
                df = pd.read_csv(uploaded_csv)
                st.write(f"已加载 {len(df)} 个分子")

                # 列名识别
                cols = df.columns.tolist()
                smiles_col = st.selectbox("选择 SMILES 列", cols, index=cols.index('smiles') if 'smiles' in cols else 0)

                if st.button("运行批量筛选"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 1. 计算 Ro5
                    status_text.text("正在计算 ADME 属性...")
                    progress_bar.progress(30)

                    # 使用 apply 进行批量计算
                    ro5_data = df[smiles_col].apply(adme_tool.calculate_ro5_properties)
                    df_result = pd.concat([df, ro5_data], axis=1)

                    # 2. 结构筛选
                    status_text.text("正在扫描不良子结构 (PAINS/Brenk)...")
                    progress_bar.progress(60)

                    df_clean, df_full_labeled, n_pains, n_brenk = struct_tool.filter_dataframe(df_result, smiles_col)

                    progress_bar.progress(100)
                    status_text.text("✅ 筛选完成")

                    # --- 结果展示 ---
                    st.divider()
                    col_stat1, col_stat2, col_stat3 = st.columns(3)

                    # 统计卡片
                    total = len(df)
                    pass_ro5 = df_result['Pass_Ro5'].sum()
                    pass_all = len(df_clean)

                    col_stat1.metric("初始分子数", total)
                    col_stat2.metric("通过 Ro5", f"{pass_ro5} ({pass_ro5/total*100:.1f}%)")
                    col_stat3.metric("最终通过筛选", f"{pass_all} ({pass_all/total*100:.1f}%)")

                    # 详细图表
                    st.subheader("📊 筛选分析报告")

                    viz_col1, viz_col2 = st.columns(2)

                    with viz_col1:
                        st.markdown("**物理化学空间分布 (通过分子)**")
                        # 绘制雷达图
                        clean_stats = df_clean[["MW", "HBA", "HBD", "LogP"]].describe().T
                        fig_radar = adme_tool.plot_radar_chart(clean_stats, "Filtered Candidates Profile")
                        if fig_radar:
                            st.pyplot(fig_radar)

                    with viz_col2:
                        st.markdown("**淘汰原因统计**")
                        # 简单的柱状图
                        reasons = {
                            "违反 Ro5": total - pass_ro5,
                            "含 PAINS": n_pains,
                            "含 Brenk": n_brenk
                        }
                        st.bar_chart(pd.Series(reasons))

                    # 数据下载
                    st.subheader("📥 结果下载")

                    tab_clean, tab_full = st.tabs(["✅ 通过筛选的分子", "📑 完整带标注数据"])

                    with tab_clean:
                        st.dataframe(df_clean.head())
                        st.download_button(
                            "下载筛选后的分子 (CSV)",
                            df_clean.to_csv(index=False).encode('utf-8'),
                            "filtered_clean_molecules.csv",
                            "text/csv"
                        )

                    with tab_full:
                        st.dataframe(df_full_labeled.head())
                        st.download_button(
                            "下载完整报告 (CSV)",
                            df_full_labeled.to_csv(index=False).encode('utf-8'),
                            "full_screening_report.csv",
                            "text/csv"
                        )

with tab3:
    st.header("🔍 化学依据分析")
    st.caption("计算分子理化性质（LogP、分子量等）、基于 Morgan 指纹的相似性搜索，以及多种分子表示对比。")
    st.info(
        "🎓 **教学点**：学习分子描述符（如LogP、TPSA）和分子指纹（Morgan指纹）如何量化分子特性，"
        "并通过相似性搜索发现已知活性化合物。"
    )
    if CHEM_INSIGHT_AVAILABLE:
        render_safe_chem_insight()
    else:
        st.error("化学洞察模块不可用")
        st.code("请确保 chem_insight_safe.py 和 molecule_utils.py 文件存在")

with tab4:
    st.header("🎯 药效团设计")
    st.caption("从活性分子中提取共同药效团特征（氢键供/受体、疏水区等），生成 3D 药效团模型，指导分子优化。")
    st.info(
        "🎓 **教学点**：从多个活性分子中提取共同药效团特征（氢键供/受体、疏水区、芳香环），"
        "建立3D药效团模型，理解\"哪些原子团对活性至关重要\"。"
    )
    if PHARMACOPHORE_AVAILABLE:
        pharmacophore_streamlit.render_pharmacophore_tab()
    else:
        st.error("药效团模块不可用")
        st.code("请确保 pharmacophore_streamlit.py 文件存在")

with tab5:
    st.header("🔗 蛋白质-配体 3D 结构可视化")
    st.caption("加载蛋白质-配体复合物（PDB ID 或本地文件），交互式查看三维结构及相互作用。")
    st.info(
        "🎓 **教学点**：观察蛋白质-配体复合物的三维结构，理解相互作用（氢键、疏水作用）如何影响结合亲和力。"
        "可加载EGFR相关PDB结构（如3POZ、1M17）。"
    )

    if not VIZ_AVAILABLE:
        st.error("⚠️ 可视化模块加载失败")
        st.code(f"错误详情: {VIZ_ERROR}", language="text")
        st.info("请根据上方错误详情检查：\n1. requirements.txt 是否安装成功\n2. structure_viz.py 文件是否存在\n3. 代码是否有语法错误")
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

            # ========== 新增：刷新控制功能 ==========
            st.markdown("---")
            st.subheader("3. 刷新控制")
            
            # 定义一个 session state 来存储"实际渲染"的参数
            if 'render_params' not in st.session_state:
                st.session_state.render_params = {
                    'style': 'cartoon', 'color': 'spectrum', 
                    'ligand': True, 'surface': False, 'opacity': 0.5
                }

            # 暂停开关
            pause_refresh = st.toggle("⏸️ 暂停实时刷新", value=False, help="开启后，修改上方样式不会立即触发重绘，需点击'手动刷新'按钮。")
            
            do_update = False
            
            if pause_refresh:
                # 暂停模式：只有点击按钮才更新
                if st.button("🔄 手动刷新视图", type="primary", use_container_width=True):
                    do_update = True
                else:
                    st.caption("⚠️ 视图已锁定，修改样式后请点击上方按钮更新。")
            else:
                # 实时模式：只要参数变了就更新
                do_update = True

            # 决定最终传给渲染器的参数
            if do_update:
                st.session_state.render_params = {
                    'style': style_select,
                    'color': color_select,
                    'ligand': show_ligand,
                    'surface': show_surface,
                    'opacity': surface_opacity
                }
            
            # 获取当前用于渲染的参数（可能是旧的，也可能是新的）
            current_render = st.session_state.render_params

        # --- 右侧显示区 ---
        with col_view:
            if st.session_state.viz_data_loaded:
                # 获取当前的 PDB 数据字符串
                # 注意：我们使用 session_state 中存储的原始字符串，确保传递给缓存函数的是不可变数据
                current_pdb_data = st.session_state.get('viz_raw_data')
                current_pdb_id = st.session_state.get('viz_pdb_id', 'Unknown')
                
                st.info(f"正在查看: **{current_pdb_id}**")
                
                # 生成视图
                try:
                    # ================== 修复代码 ==================
                    # 调用缓存函数，而不是直接调用 viz_tool.render_view
                    # 使用 current_render 中的参数，而不是 widget 变量
                    view = get_3d_view(
                        pdb_data=current_pdb_data,
                        style=current_render['style'],
                        color_scheme=current_render['color'],
                        show_ligand=current_render['ligand'],
                        show_surface=current_render['surface'],
                        surface_opacity=current_render['opacity']
                    )
                    # ============================================
                    
                    # 在 Streamlit 中显示
                    if view:
                        showmol(view, height=600, width=800)
                    else:
                        st.error("视图生成失败")
                    
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
    st.header("📊 模型性能分析")
    st.caption("查看双引擎模型的性能指标（AUC、准确率）、特征重要性排序和混淆矩阵。")
    st.info(
        "🎓 **教学点**：查看双引擎模型的性能指标（AUC、准确率）和特征重要性，"
        "理解模型评估方法及可解释性分析的价值。"
    )

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

with tab7:
    st.header("🔬 技术实现详情")
    st.caption("双引擎架构、技术栈与模型性能一览")
    st.info(
        "🎓 **教学点**：了解系统架构、技术栈和特征工程对比，深入理解AI药物设计平台的技术实现。"
    )

    st.markdown("""
    ### 🏗️ 双引擎架构

    ```
    输入层 (SMILES)
        ├── 随机森林分支 → RDKit特征提取 (200+描述符) → 预测结果
        └── GNN分支 → 分子图转换 (12维原子特征) → 图卷积网络 → 预测结果
    ```

    - **随机森林 (RF)**：基于化学经验的全局特征学习（AUC 0.855）
    - **图神经网络 (GNN)**：基于分子拓扑的局部结构感知（AUC 0.808）
    - **集成决策**：加权平均 + 一致性判断，提升可靠性

    ### 🔧 核心技术栈

    | 组件 | 技术选型 | 用途 |
    |------|----------|------|
    | Web框架 | Streamlit | 交互式界面 |
    | 机器学习 | scikit-learn | 随机森林模型 |
    | 深度学习 | PyTorch + PyTorch Geometric | GNN模型 |
    | 化学信息学 | RDKit | 分子特征与可视化 |
    | 3D可视化 | py3Dmol + stmol | 蛋白-配体结构 |

    ### 📊 模型性能对比

    | 指标 | 随机森林 | GNN | 说明 |
    |------|----------|-----|------|
    | AUC | 0.855 | 0.808 | 分类性能 |
    | 准确率 | 83.0% | 76.5% | 测试集结果 |
    | 可解释性 | 高（特征重要性） | 中（图注意力） | 教学价值 |

    ### 🎯 教学价值
    - **对比学习**：直观比较传统特征工程与深度学习在药物发现中的表现
    - **可解释性**：RF特征重要性揭示活性关键因素（如LogP、芳香环数）
    - **端到端体验**：从SMILES输入到3D结构展示，完整CADD流程触手可及
    """)

with tab8:
    st.header("📚 关于药尘光")
    st.caption("项目背景、核心理念、特色与致谢")
    st.info(
        "🎓 **教学点**：了解项目背景、特色、数据来源及开源资源，培养科研诚信与可复现意识。"
    )

    st.markdown("""
    ### 🎯 项目简介

    **药尘光** 是一款面向 EGFR 抑制剂智能发现的 **教学友好型 Web 平台**，
    致力于将前沿 AI 技术（随机森林 + 图神经网络）转化为本科生触手可及的交互式学习工具。

    > *"双核驱动，理形相生"*
    > —— 随机森林捕捉"经验之理"，图神经网络感知"结构之形"，双引擎相互验证，让 AI 决策透明可解释。

    ### 🌟 核心特色

    - **🧪 双引擎预测**：RF (AUC 0.855) + GNN (AUC 0.808)，对比学习两种 AI 范式
    - **🎓 教学优先**：渐进式标签页 + 可解释性输出 + 实时引导，零基础上手
    - **☁️ 云端即用**：无需安装，浏览器访问 https://ai-egfr-platform.streamlit.app/
    - **📖 开源共享**：代码完全开源，数据源自 ChEMBL，支持二次开发与教学复用

    ### 📦 资源与致谢

    - **数据来源**：ChEMBL 数据库（5,568 个 EGFR 化合物）
    - **技术框架**：Streamlit、RDKit、PyTorch Geometric、scikit-learn
    - **开源协议**：仅供学术研究使用，详情见 GitHub 仓库

    ---
    **GitHub**：https://github.com/d7ftjy8n4j-cell/ai-egfr-platform
    **反馈建议**：欢迎提交 Issue 或 Pull Request
    """)

# ========== 6. 侧边栏信息 ==========
with st.sidebar:
    # 品牌区
    st.markdown("## **药尘光**")
    st.caption("*双核驱动，理形相生*")
    st.divider()

    # 原有的系统配置等代码保持不变...
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

    # ----- 新增：教学指南 -----
    with st.expander("📘 教学指南（新手必读）", expanded=False):
        st.markdown("""
        **药尘光 · 学习路径**  
        1. **🧪 分子预测**：输入SMILES，体验双引擎对比  
        2. **🛡️ 药物筛选**：评估成药性与毒性风险  
        3. **🔍 化学依据**：探索分子性质与相似性  
        4. **🎯 药效团设计**：生成3D药效团模型  
        5. **🔗 3D结构**：观察蛋白-配体相互作用  
        6. **📊 模型分析**：理解模型性能与特征  
        ---
        *"双核驱动，理形相生"*  
        随机森林（理）与图神经网络（形）相互验证，让AI决策透明可解释。
        """)
    # ---------------------------

    # ----- 新增：功能导航指南 -----
    with st.expander("📖 功能导航指南", expanded=False):
        st.markdown("""
        - **🧪 分子预测**：核心活性预测，支持单分子/批量
        - **🛡️ 药物筛选**：成药性评估（Lipinski）与毒性警报（PAINS/Brenk）
        - **🔍 化学依据**：分子性质计算、相似性搜索、表示对比
        - **🎯 药效团设计**：提取活性特征，生成 3D 药效团模型
        - **🔗 3D 结构**：蛋白-配体相互作用可视化
        - **📊 模型分析**：模型性能、特征重要性、混淆矩阵
        - **🔬 技术详情**：系统架构、技术栈、特征工程对比
        - **📚 关于项目**：背景、特色、文件清单、致谢
        """)
    # ---------------------------

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
    🧬 药尘光 · EGFR抑制剂智能发现平台 | 双核驱动，理形相生 | © 2026
    <br>
    <small>面向本科生的AIDD教学平台 · 打开浏览器即学即用</small>
    </div>
    """,
    unsafe_allow_html=True
)