"""
chem_insight_safe.py - 安全的化学洞察模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import logging
from pathlib import Path

# 导入安全工具
try:
    from molecule_utils import SafeMolecule
    SAFE_MOL_AVAILABLE = True
except ImportError:
    st.error("❌ 缺少molecule_utils模块")
    SAFE_MOL_AVAILABLE = False

# 导入RDKit
try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem, Draw, rdMolDescriptors, Descriptors, Lipinski, Crippen
    RDKIT_AVAILABLE = True
except ImportError as e:
    RDKIT_AVAILABLE = False
    st.error(f"❌ RDKit导入失败: {e}")

class SafeChemInsightEngine:
    """安全的化学洞察引擎"""
    
    def __init__(self, reference_data_path=None):
        self.reference_df = None
        self.data_mode = "offline"
        
        if not RDKIT_AVAILABLE or not SAFE_MOL_AVAILABLE:
            st.warning("⚠️ 依赖库不可用，化学洞察功能受限")
            return
        
        # 加载数据
        if reference_data_path and Path(reference_data_path).exists():
            self._load_data_from_path(reference_data_path)
        else:
            self._load_offline_data()
    
    def _load_data_from_path(self, data_path):
        """从路径加载数据"""
        try:
            self.reference_df = pd.read_csv(data_path)
            if 'smiles' not in self.reference_df.columns:
                st.error(f"数据文件缺少'smiles'列: {data_path}")
                self._load_offline_data()
                return
            
            # 验证所有SMILES
            valid_count = 0
            for idx, row in self.reference_df.iterrows():
                smiles = str(row['smiles'])
                is_valid, mol, _ = SafeMolecule.validate_smiles(smiles)
                if is_valid:
                    valid_count += 1
            
            st.sidebar.success(f"✅ 参考数据集已加载: {valid_count}/{len(self.reference_df)} 个有效分子")
            self.data_mode = "uploaded"
            
        except Exception as e:
            st.warning(f"⚠️ 无法加载参考数据集: {e}")
            self._load_offline_data()
    
    def _load_offline_data(self):
        """加载离线示例数据"""
        offline_examples = [
            {'smiles': 'Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1', 'activity': 8.2, 'name': 'EGFR核心骨架'},
            {'smiles': 'COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4', 'activity': 7.9, 'name': '吉非替尼类似物'},
            {'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'activity': 4.5, 'name': '咖啡因（阴性对照）'},
            {'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O', 'activity': 3.8, 'name': '阿司匹林（阴性对照）'},
            {'smiles': 'C1=CC=C(C=C1)C=O', 'activity': 5.2, 'name': '苯甲醛'},
        ]
        self.reference_df = pd.DataFrame(offline_examples)
        self.data_mode = "offline"
        
    
    def safe_find_similar_compounds(self, query_smiles, top_n=5):
        """安全地查找相似化合物"""
        if not RDKIT_AVAILABLE or self.reference_df is None:
            return []
        
        # 验证查询分子
        is_valid, query_mol, error_msg = SafeMolecule.validate_smiles(query_smiles)
        if not is_valid:
            st.error(f"查询分子无效: {error_msg}")
            return []
        
        results = []
        
        for _, row in self.reference_df.iterrows():
            ref_smiles = str(row.get('smiles', ''))
            if not ref_smiles:
                continue
            
            # 验证参考分子
            ref_valid, ref_mol, _ = SafeMolecule.validate_smiles(ref_smiles)
            if not ref_valid:
                continue
            
            try:
                # 计算相似度
                fp1 = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                
                # 获取活性值
                activity = row.get('activity')
                if pd.isna(activity):
                    activity = None
                
                results.append({
                    'smiles': ref_smiles,
                    'similarity': similarity,
                    'mol_obj': ref_mol,
                    'activity': activity,
                    'name': row.get('name', ''),
                    'is_active': activity is not None and float(activity) > 6.5
                })
            except Exception as e:
                logging.debug(f"相似度计算跳过: {e}")
                continue
        
        # 排序并返回
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_n]
    
    def safe_calculate_properties(self, smiles):
        """安全地计算分子性质"""
        is_valid, mol, error_msg = SafeMolecule.validate_smiles(smiles)
        
        if not is_valid:
            return {
                'error': error_msg,
                'is_valid': False
            }
        
        properties = SafeMolecule.safe_calculate_properties(mol)
        properties['is_valid'] = True
        properties['atom_count'] = mol.GetNumAtoms()
        
        return properties
    
    def safe_visualize_molecule(self, smiles, figsize=(12, 8)):
        """安全地可视化分子"""
        is_valid, mol, error_msg = SafeMolecule.validate_smiles(smiles)
        
        if not is_valid:
            st.error(f"无法可视化分子: {error_msg}")
            return None
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            fig.suptitle(f'分子分析: {smiles[:40]}{"..." if len(smiles) > 40 else ""}', fontsize=14)
            
            # 1. 2D结构
            ax = axes[0, 0]
            img = Draw.MolToImage(mol, size=(300, 300))
            ax.imshow(img)
            ax.set_title('2D分子结构')
            ax.axis('off')
            
            # 2. 分子量
            ax = axes[0, 1]
            mw = Descriptors.ExactMolWt(mol)
            ax.bar(['分子量'], [mw], color='skyblue')
            ax.set_ylabel('Da')
            ax.set_title(f'分子量: {mw:.1f}')
            
            # 3. 原子组成
            ax = axes[0, 2]
            atom_counts = {}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
            
            bars = ax.bar(list(atom_counts.keys()), list(atom_counts.values()), 
                         color='lightgreen')
            ax.set_title('原子组成')
            ax.set_ylabel('数量')
            
            # 在柱子上添加数值
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            # 4. 指纹热图
            ax = axes[1, 0]
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
            fp_array = np.zeros((1, len(fp)))
            DataStructs.ConvertToNumpyArray(fp, fp_array[0])
            fp_vis = fp_array[0, :64].reshape(8, 8)
            im = ax.imshow(fp_vis, cmap='Blues', aspect='auto')
            ax.set_title('Morgan指纹 (前64位)')
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # 5. 药效团特征
            ax = axes[1, 1]
            features = []
            values = []
            
            # 计算一些简单的药效团特征
            try:
                hbd = Lipinski.NumHDonors(mol)
                features.append('HBD')
                values.append(hbd)
            except:
                pass
            
            try:
                hba = Lipinski.NumHAcceptors(mol)
                features.append('HBA')
                values.append(hba)
            except:
                pass
            
            try:
                rings = rdMolDescriptors.CalcNumAromaticRings(mol)
                features.append('芳香环')
                values.append(rings)
            except:
                pass
            
            if features:
                bars = ax.bar(features, values, color=['#FF9999', '#66B2FF', '#99FF99'])
                ax.set_title('药效团特征')
                ax.set_ylabel('数量')
                
                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{val}', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, '药效团特征\n计算失败', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            # 6. 相似度说明
            ax = axes[1, 2]
            ax.text(0.5, 0.7, '化学相似性分析', 
                   ha='center', va='center', fontsize=12, weight='bold')
            ax.text(0.5, 0.5, '• Morgan指纹: 拓扑相似度\n• 原子组成: 元素分析\n• 2D结构: 可视化验证', 
                   ha='center', va='center', fontsize=10)
            ax.text(0.5, 0.3, 'T004 + T033集成', 
                   ha='center', va='center', fontsize=9, style='italic')
            ax.axis('off')
            ax.set_title('分析原理')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logging.error(f"可视化失败: {e}")
            return None

def render_safe_chem_insight():
    """渲染安全的化学洞察页面"""
    
    if not RDKIT_AVAILABLE:
        st.error("""
        ❌ RDKit库不可用
        
        请在终端中运行以下命令安装:
        ```
        pip install rdkit
        ```
        或使用conda:
        ```
        conda install -c conda-forge rdkit
        ```
        """)
        return
    
    
    st.markdown("""
    **集成TeachOpenCADD T004(化合物相似性) + T033(分子表示方法)**
    - 🧬 基于化学相似性的活性验证
    - 📊 多维度分子表示对比
    - 🛡️ 安全的错误处理机制
    """)
    
    # 初始化引擎
    engine = SafeChemInsightEngine()
    
    # 获取要分析的分子
    default_smiles = "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1"
    query_smiles = st.session_state.get('last_smiles', default_smiles)
    
    # 输入框
    query_smiles = st.text_input(
        "输入SMILES字符串进行分析:",
        value=query_smiles,
        help="输入有效的SMILES字符串，如: CCO (乙醇)"
    )
    
    if not query_smiles:
        st.warning("请输入SMILES字符串")
        return
    
    # 验证分子
    is_valid, mol, error_msg = SafeMolecule.validate_smiles(query_smiles)
    
    if not is_valid:
        st.error(f"❌ 无效的SMILES: {error_msg}")
        st.info("""
        **常见SMILES格式问题:**
        1. 确保使用标准SMILES格式
        2. 检查括号是否匹配
        3. 确保原子符号正确
        4. 示例有效SMILES: `CCO`, `CC(=O)O`, `c1ccccc1`
        """)
        return
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["🧬 相似性分析", "📊 分子表示", "📈 性质计算"])
    
    with tab1:
        st.subheader("相似化合物分析")
        
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("显示相似化合物数量:", 3, 10, 5)
        
        if st.button("🔍 查找相似化合物", type="primary"):
            with st.spinner("正在搜索相似化合物..."):
                similar_mols = engine.safe_find_similar_compounds(query_smiles, top_n=top_n)
                
                if similar_mols:
                    st.success(f"✅ 找到 {len(similar_mols)} 个相似化合物")
                    
                    # 显示统计信息
                    avg_sim = np.mean([m['similarity'] for m in similar_mols])
                    active_count = sum([1 for m in similar_mols if m.get('is_active', False)])
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    col_stat1.metric("平均相似度", f"{avg_sim:.3f}")
                    col_stat2.metric("活性化合物", active_count)
                    col_stat3.metric("最高相似度", 
                                   f"{max([m['similarity'] for m in similar_mols]):.3f}")
                    
                    # 显示相似分子
                    st.subheader("相似化合物详情")
                    
                    # 使用列显示
                    cols = st.columns(min(5, len(similar_mols)))
                    for idx, (col, mol_info) in enumerate(zip(cols, similar_mols)):
                        with col:
                            if mol_info['mol_obj']:
                                img = Draw.MolToImage(mol_info['mol_obj'], size=(150, 100))
                                col.image(img, 
                                         caption=f"#{idx+1}: {mol_info['similarity']:.3f}")
                            
                            # 活性状态
                            activity = mol_info.get('activity')
                            if activity is not None:
                                activity_text = f"pIC50: {activity}"
                                if mol_info.get('is_active', False):
                                    col.success(f"✅ {activity_text}")
                                else:
                                    col.error(f"❌ {activity_text}")
                            else:
                                col.info("活性: N/A")
                            
                            # SMILES片段
                            smiles_short = mol_info['smiles'][:30] + ("..." if len(mol_info['smiles']) > 30 else "")
                            col.caption(f"`{smiles_short}`")
                    
                    # 化学意义解读
                    st.subheader("🧪 化学意义解读")
                    
                    best_match = similar_mols[0]
                    if best_match['similarity'] > 0.7:
                        st.success("""
                        **高相似度发现**: 查询分子与已知化合物高度相似，预测结果可靠性高。
                        """)
                        if best_match.get('is_active', False):
                            st.info(f"最相似分子为**活性化合物**(pIC50={best_match.get('activity', 'N/A')})，支持活性预测。")
                    elif best_match['similarity'] > 0.4:
                        st.warning("""
                        **中等相似度**: 分子具有一定结构相似性，但存在差异。
                        """)
                    else:
                        st.info("""
                        **低相似度**: 未找到高度相似化合物，预测主要基于模型学习能力。
                        """)
                    
                else:
                    st.warning("未找到相似化合物")
    
    with tab2:
        st.subheader("分子表示方法对比")
        
        if st.button("🔄 生成分子分析图", type="secondary"):
            with st.spinner("正在生成分子分析..."):
                fig = engine.safe_visualize_molecule(query_smiles)
                
                if fig:
                    st.pyplot(fig)
                    
                    # 提供下载
                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 下载分析图",
                        data=buf,
                        file_name="molecular_analysis.png",
                        mime="image/png"
                    )
                else:
                    st.error("无法生成分子分析图")
    
    with tab3:
        st.subheader("分子性质计算")
        
        properties = engine.safe_calculate_properties(query_smiles)
        
        if properties.get('error'):
            st.error(f"性质计算失败: {properties.get('error')}")
        else:
            # 显示性质卡片
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("分子量", f"{properties.get('mw', 0):.1f} Da")
                st.metric("LogP", f"{properties.get('logp', 0):.2f}")
                st.metric("氢键供体", properties.get('hbd', 0))
            
            with col2:
                st.metric("氢键受体", properties.get('hba', 0))
                st.metric("可旋转键", properties.get('rotatable_bonds', 0))
                st.metric("芳香环", properties.get('aromatic_rings', 0))
            
            with col3:
                st.metric("拓扑极性表面积", f"{properties.get('tpsa', 0):.1f} Å²")
                st.metric("重原子数", properties.get('heavy_atoms', 0))
                st.metric("形式电荷", properties.get('formal_charge', 0))
            
            # 性质解读
            st.subheader("🧪 性质解读")
            
            # Lipinski五规则检查
            rules = [
                ("分子量 ≤ 500", properties.get('mw', 0) <= 500),
                ("LogP ≤ 5", properties.get('logp', 0) <= 5),
                ("氢键供体 ≤ 5", properties.get('hbd', 0) <= 5),
                ("氢键受体 ≤ 10", properties.get('hba', 0) <= 10),
            ]
            
            passed = sum([1 for _, condition in rules if condition])
            
            if passed >= 4:
                st.success(f"✅ 符合Lipinski五规则 ({passed}/4)")
            elif passed >= 3:
                st.warning(f"⚠️ 部分符合Lipinski五规则 ({passed}/4)")
            else:
                st.error(f"❌ 不符合Lipinski五规则 ({passed}/4)")
            
            # 详细规则检查
            for rule_name, condition in rules:
                if condition:
                    st.info(f"✓ {rule_name}")
                else:
                    st.warning(f"✗ {rule_name}")

# 主函数
def main():
    st.set_page_config(
        page_title="化学洞察 - TeachOpenCADD T004+T033",
        page_icon="🔍",
        layout="wide"
    )
    render_safe_chem_insight()

if __name__ == "__main__":
    main()