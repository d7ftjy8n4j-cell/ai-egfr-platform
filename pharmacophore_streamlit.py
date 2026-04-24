"""
pharmacophore_streamlit.py - Streamlit集成的药效团生成模块
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import base64
from io import BytesIO
import json
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
# rdShapeHelpers 在某些RDKit版本中可能不可用，尝试导入但不强制要求
try:
    from rdkit.Chem import rdShapeHelpers
except ImportError:
    rdShapeHelpers = None
import py3Dmol
# stmol 在Streamlit中可能有问题，使其可选导入
try:
    from stmol import showmol
    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False
    showmol = None
import warnings
warnings.filterwarnings('ignore')

class StreamlitPharmacophore:
    """Streamlit友好的药效团生成器"""
    
    def __init__(self):
        self.feature_colors = {
            "donor": (0, 1, 0),      # 绿色
            "acceptor": (1, 0, 0),   # 红色
            "hydrophobe": (1, 1, 0), # 黄色
            "positive": (0, 0, 1),   # 蓝色
            "negative": (1, 0, 1),   # 紫色
            "aromatic": (1, 0.5, 0)  # 橙色
        }
        
        self.feature_radii = {
            "donor": 1.5,
            "acceptor": 1.5,
            "hydrophobe": 2.0,
            "positive": 1.5,
            "negative": 1.5,
            "aromatic": 1.8
        }
        
        self.molecules = []
        self.molecule_names = []
        self.features = []
        
    def load_molecules_from_smiles(self, smiles_list, names=None):
        """从SMILES列表加载分子并生成3D构象"""
        self.molecules = []
        self.molecule_names = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    st.warning(f"无法解析SMILES: {smiles}")
                    continue
                    
                # 添加氢原子
                mol = Chem.AddHs(mol)
                
                # 生成3D构象
                AllChem.EmbedMolecule(mol, randomSeed=42+i)
                AllChem.MMFFOptimizeMolecule(mol)
                
                self.molecules.append(mol)
                
                if names and i < len(names):
                    self.molecule_names.append(names[i])
                else:
                    self.molecule_names.append(f"分子_{i+1}")
                    
            except Exception as e:
                st.error(f"分子 {i+1} 处理失败: {str(e)[:100]}")
                
        return len(self.molecules)
    
    def extract_pharmacophore_features(self):
        """提取药效团特征（简化版）"""
        if not self.molecules:
            return []
            
        self.features = []
        
        for mol in self.molecules:
            mol_features = self._extract_molecule_features(mol)
            self.features.append(mol_features)
            
        return self.features
    
    def _extract_molecule_features(self, mol):
        """提取单个分子的药效团特征"""
        features = []
        
        # 1. 氢键供体
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7 or atom.GetAtomicNum() == 8:  # N或O
                if atom.GetTotalNumHs() > 0:
                    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    features.append({
                        "type": "donor",
                        "atom_idx": atom.GetIdx(),
                        "position": [pos.x, pos.y, pos.z],
                        "strength": 1.0
                    })
        
        # 2. 氢键受体
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7 or atom.GetAtomicNum() == 8:  # N或O
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                features.append({
                    "type": "acceptor",
                    "atom_idx": atom.GetIdx(),
                    "position": [pos.x, pos.y, pos.z],
                    "strength": 1.0
                })
        
        # 3. 疏水区域（基于碳原子）
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetIsAromatic():
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                features.append({
                    "type": "aromatic",
                    "atom_idx": atom.GetIdx(),
                    "position": [pos.x, pos.y, pos.z],
                    "strength": 1.0
                })
            elif atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() >= 2:
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                features.append({
                    "type": "hydrophobe",
                    "atom_idx": atom.GetIdx(),
                    "position": [pos.x, pos.y, pos.z],
                    "strength": 0.7
                })
                
        return features
    
    def generate_ensemble_pharmacophore(self, threshold=0.7):
        """生成集成药效团模型"""
        if not self.features or len(self.features) < 2:
            return None
            
        # 收集所有特征
        all_features = []
        for mol_idx, mol_features in enumerate(self.features):
            for feature in mol_features:
                feature["mol_idx"] = mol_idx
                all_features.append(feature)
        
        # 按特征类型分组
        feature_by_type = {}
        for feature in all_features:
            ftype = feature["type"]
            if ftype not in feature_by_type:
                feature_by_type[ftype] = []
            feature_by_type[ftype].append(feature)
        
        # 对每个特征类型进行聚类（简化版：基于距离）
        ensemble_features = {}
        
        for ftype, features in feature_by_type.items():
            if len(features) < 2:
                continue
                
            # 计算特征点之间的平均位置
            positions = np.array([f["position"] for f in features])
            avg_position = np.mean(positions, axis=0)
            
            # 计算重要性（出现的分子比例）
            importance = len(set([f["mol_idx"] for f in features])) / len(self.molecules)
            
            if importance >= threshold:
                ensemble_features[ftype] = {
                    "position": avg_position.tolist(),
                    "importance": importance,
                    "count": len(features),
                    "radius": self.feature_radii.get(ftype, 1.5)
                }
                
        return ensemble_features
    
    def visualize_ensemble_pharmacophore_3d(self, ensemble_features, width=900, height=700):
        """
        3D可视化集成药效团，并自动调整缩放至合适大小
        """
        viewer = py3Dmol.view(width=width, height=height)

        # 添加所有分子（沿用之前的样式）
        for i, mol in enumerate(self.molecules):
            pdb_block = Chem.MolToPDBBlock(mol)
            viewer.addModel(pdb_block, 'pdb')
            viewer.setStyle({'model': i}, {'stick': {'colorscheme': 'grayCarbon', 'radius': 0.2}})

        # 添加药效团特征
        for ftype, feature in ensemble_features.items():
            pos = feature["position"]
            color = self.feature_colors.get(ftype, (1, 1, 1))
            radius = feature.get("radius", 1.5)
            importance = feature.get("importance", 1.0)
            opacity = 0.3 + 0.7 * importance
            viewer.addSphere({
                'center': {'x': pos[0], 'y': pos[1], 'z': pos[2]},
                'radius': radius,
                'color': f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})',
                'opacity': opacity
            })
            viewer.addLabel(
                f"{ftype}\n{importance:.0%}",
                {
                    'position': {'x': pos[0], 'y': pos[1] + radius, 'z': pos[2]},
                    'backgroundColor': f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.7)',
                    'fontColor': 'black',
                    'fontSize': 12
                }
            )

        # 关键：自动计算合适的缩放并拉近视角
        viewer.zoomTo()           # 让相机对准所有对象
        viewer.zoom(1.6)          # 额外放大60%

        return viewer
    
    def visualize_2d_pharmacophore(self, ensemble_features, width=600, height=400):
        """2D可视化药效团（Matplotlib）"""
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        # 创建极坐标图显示特征分布
        feature_types = list(ensemble_features.keys())
        importances = [ensemble_features[ft]["importance"] for ft in feature_types]
        counts = [ensemble_features[ft]["count"] for ft in feature_types]
        
        # 创建颜色
        colors = [self.feature_colors.get(ft, (0.5, 0.5, 0.5)) for ft in feature_types]
        
        # 创建条形图
        y_pos = np.arange(len(feature_types))
        
        # 重要性条形
        bars1 = ax.barh(y_pos - 0.2, importances, 0.4, 
                       color=colors, alpha=0.6, label='重要性')
        
        # 数量条形
        bars2 = ax.barh(y_pos + 0.2, [c/max(counts) for c in counts], 0.4,
                       color=colors, alpha=0.3, label='数量(归一化)')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_types)
        ax.set_xlabel('值')
        ax.set_title('药效团特征统计')
        ax.legend()
        
        # 添加数值标签
        for i, (imp, cnt) in enumerate(zip(importances, counts)):
            ax.text(imp, i-0.2, f'{imp:.0%}', va='center', ha='left')
            ax.text(cnt/max(counts), i+0.2, f'{cnt}', va='center', ha='left')
        
        plt.tight_layout()
        return fig
    
    def generate_pharmacophore_report(self, ensemble_features):
        """生成药效团分析报告"""
        report = {
            "summary": {
                "total_molecules": len(self.molecules),
                "total_features": len(ensemble_features),
                "average_importance": np.mean([f["importance"] for f in ensemble_features.values()])
            },
            "features": ensemble_features,
            "molecules": [
                {
                    "name": name,
                    "smiles": Chem.MolToSmiles(mol),
                    "num_atoms": mol.GetNumAtoms(),
                    "mw": rdMolDescriptors.CalcExactMolWt(mol)
                }
                for name, mol in zip(self.molecule_names, self.molecules)
            ]
        }
        return report
    
    def save_pharmacophore_model(self, ensemble_features, filepath):
        """保存药效团模型为JSON文件"""
        with open(filepath, 'w') as f:
            json.dump({
                "features": ensemble_features,
                "molecules": [
                    {
                        "name": name,
                        "smiles": Chem.MolToSmiles(mol)
                    }
                    for name, mol in zip(self.molecule_names, self.molecules)
                ],
                "metadata": {
                    "generator": "StreamlitPharmacophore",
                    "version": "1.0",
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            }, f, indent=2)
        
        return filepath

def render_pharmacophore_tab():
    """渲染药效团生成标签页"""
    
    
    st.markdown("""
    从已知活性分子生成集成药效团模型，用于指导分子设计和优化。
    **功能**: 分子加载 → 特征提取 → 聚类 → 药效团生成 → 3D可视化
    """)
    
    # 初始化药效团生成器
    if 'pharmacophore_generator' not in st.session_state:
        st.session_state.pharmacophore_generator = StreamlitPharmacophore()
    
    generator = st.session_state.pharmacophore_generator
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 数据输入",
        "🔍 特征分析",
        "🎯 药效团生成",
        "💾 导出结果"
    ])
    
    with tab1:
        st.subheader("输入活性分子")
        
        input_method = st.radio(
            "选择输入方式:",
            ["📝 手动输入SMILES", "📁 上传文件", "🔗 使用示例数据"]
        )
        
        if input_method == "📝 手动输入SMILES":
            smiles_input = st.text_area(
                "输入SMILES（每行一个分子）:",
                height=200,
                value="""Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1
COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
CC(=O)OC1=CC=CC=C1C(=O)O"""
            )
            
            names_input = st.text_area(
                "输入分子名称（可选，每行一个）:",
                height=100,
                value="""EGFR抑制剂1
吉非替尼类似物
咖啡因
阿司匹林"""
            )
            
            if st.button("🚀 加载分子", type="primary"):
                smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
                names_list = [n.strip() for n in names_input.split('\n') if n.strip()]
                
                with st.spinner(f"正在加载 {len(smiles_list)} 个分子..."):
                    loaded_count = generator.load_molecules_from_smiles(smiles_list, names_list)
                    st.success(f"✅ 成功加载 {loaded_count} 个分子")
                    
                    # 显示加载的分子
                    if loaded_count > 0:
                        st.subheader("加载的分子预览")
                        cols = st.columns(min(4, loaded_count))
                        for idx, (col, mol) in enumerate(zip(cols, generator.molecules)):
                            with col:
                                img = Draw.MolToImage(mol, size=(200, 200))
                                col.image(img, caption=generator.molecule_names[idx])
        
        elif input_method == "📁 上传文件":
            uploaded_file = st.file_uploader(
                "上传分子文件 (支持 .smi, .txt, .csv)",
                type=['smi', 'txt', 'csv'],
                help="文件应包含SMILES列，可选名称列"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        smiles_col = st.selectbox("选择SMILES列:", df.columns)
                        
                        if 'name' in df.columns:
                            name_col = st.selectbox("选择名称列:", df.columns, index=list(df.columns).index('name'))
                        else:
                            name_col = None
                        
                        smiles_list = df[smiles_col].tolist()
                        names_list = df[name_col].tolist() if name_col else None
                    else:
                        content = uploaded_file.getvalue().decode()
                        lines = content.strip().split('\n')
                        smiles_list = []
                        names_list = []
                        
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                smiles_list.append(parts[0])
                                if len(parts) >= 2:
                                    names_list.append(parts[1])
                        
                    if st.button("加载上传的分子"):
                        loaded_count = generator.load_molecules_from_smiles(smiles_list, names_list)
                        st.success(f"✅ 从文件加载 {loaded_count} 个分子")
                        
                except Exception as e:
                    st.error(f"文件处理失败: {str(e)}")
        
        else:  # 使用示例数据
            st.info("使用EGFR抑制剂示例数据集")
            
            example_smiles = [
                "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",
                "COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                "C1=CC=C(C=C1)C=O"
            ]
            
            example_names = [
                "EGFR核心骨架",
                "吉非替尼类似物",
                "咖啡因（阴性对照）",
                "阿司匹林（阴性对照）",
                "苯甲醛"
            ]
            
            if st.button("加载示例数据", type="secondary"):
                with st.spinner("加载示例数据..."):
                    loaded_count = generator.load_molecules_from_smiles(example_smiles, example_names)
                    st.success(f"✅ 加载 {loaded_count} 个示例分子")
                    
                    # 显示示例分子
                    st.subheader("示例分子预览")
                    cols = st.columns(5)
                    for idx, (col, mol) in enumerate(zip(cols, generator.molecules)):
                        with col:
                            img = Draw.MolToImage(mol, size=(150, 150))
                            col.image(img, caption=example_names[idx])
    
    with tab2:
        st.subheader("药效团特征分析")
        
        if not generator.molecules:
            st.warning("请先在'数据输入'标签页加载分子")
        else:
            if st.button("🔍 提取特征", type="primary"):
                with st.spinner("正在提取药效团特征..."):
                    features = generator.extract_pharmacophore_features()
                    
                    # 显示特征统计
                    total_features = sum([len(f) for f in features])
                    feature_types = {}
                    
                    for mol_features in features:
                        for feature in mol_features:
                            ftype = feature["type"]
                            feature_types[ftype] = feature_types.get(ftype, 0) + 1
                    
                    st.success(f"✅ 提取 {total_features} 个特征")
                    
                    # 特征类型分布
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("特征类型分布")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        types = list(feature_types.keys())
                        counts = list(feature_types.values())
                        colors = [generator.feature_colors.get(t, (0.5, 0.5, 0.5)) for t in types]
                        
                        ax.bar(types, counts, color=colors)
                        ax.set_ylabel('数量')
                        ax.set_title('特征类型分布')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("特征详情")
                        feature_data = []
                        for i, mol_features in enumerate(features):
                            for feature in mol_features:
                                feature_data.append({
                                    "分子": generator.molecule_names[i],
                                    "类型": feature["type"],
                                    "X": f"{feature['position'][0]:.2f}",
                                    "Y": f"{feature['position'][1]:.2f}",
                                    "Z": f"{feature['position'][2]:.2f}"
                                })
                        
                        if feature_data:
                            df_features = pd.DataFrame(feature_data)
                            st.dataframe(df_features, use_container_width=True)
    
    with tab3:
        st.subheader("生成集成药效团")
        
        if not generator.features:
            st.warning("请先提取特征")
        else:
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                threshold = st.slider("重要性阈值:", 0.1, 1.0, 0.7, 0.05)
            
            with col_param2:
                cluster_method = st.selectbox("聚类方法:", ["距离平均", "K-means", "DBSCAN"])
            
            if st.button("🎯 生成药效团模型", type="primary"):
                with st.spinner("正在生成集成药效团..."):
                    ensemble_features = generator.generate_ensemble_pharmacophore(threshold)
                    
                    if ensemble_features:
                        st.success(f"✅ 生成 {len(ensemble_features)} 个药效团特征")

                        # 3D可视化
                        st.subheader("3D药效团模型")
                        viewer = generator.visualize_ensemble_pharmacophore_3d(
                            ensemble_features,
                            width=1000,
                            height=800
                        )

                        # 使用原生HTML嵌入（全宽显示）
                        st.components.v1.html(viewer._make_html(), height=800, width=1000, scrolling=False)
                        
                        # 2D统计图
                        st.subheader("特征统计")
                        fig_2d = generator.visualize_2d_pharmacophore(ensemble_features)
                        st.pyplot(fig_2d)
                        
                        # 特征说明
                        st.subheader("🧪 特征说明")
                        
                        feature_explanations = {
                            "donor": "氢键供体：提供氢原子形成氢键（如 -NH, -OH）",
                            "acceptor": "氢键受体：接受氢原子形成氢键（如 C=O, N:)",
                            "hydrophobe": "疏水区域：疏水相互作用区域（如脂肪链）",
                            "aromatic": "芳香环：π-π堆积作用",
                            "positive": "正电荷：静电相互作用",
                            "negative": "负电荷：静电相互作用"
                        }
                        
                        for ftype, feature in ensemble_features.items():
                            importance = feature["importance"]
                            count = feature["count"]
                            explanation = feature_explanations.get(ftype, "未知特征")
                            
                            st.info(f"""
                            **{ftype.upper()}** 
                            - 重要性: {importance:.0%}
                            - 出现次数: {count}
                            - 解释: {explanation}
                            """)
                        
                        # 保存到session state
                        st.session_state.ensemble_features = ensemble_features
                        
                    else:
                        st.error("无法生成药效团，请降低阈值或添加更多分子")
    
    with tab4:
        st.subheader("导出药效团结果")
        
        if 'ensemble_features' not in st.session_state:
            st.warning("请先生成药效团模型")
        else:
            ensemble_features = st.session_state.ensemble_features
            
            # 生成报告
            report = generator.generate_pharmacophore_report(ensemble_features)
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                st.subheader("JSON导出")
                json_str = json.dumps(report, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 下载JSON报告",
                    data=json_str,
                    file_name="pharmacophore_report.json",
                    mime="application/json"
                )
                
                # 显示报告预览
                with st.expander("预览JSON报告"):
                    st.code(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)
            
            with col_export2:
                st.subheader("图像导出")
                
                # 生成2D图像
                fig = generator.visualize_2d_pharmacophore(ensemble_features)
                
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="📥 下载统计图",
                    data=buf,
                    file_name="pharmacophore_statistics.png",
                    mime="image/png"
                )
            
            # 应用建议
            st.subheader("🎯 药物设计建议")
            
            if ensemble_features:
                suggestions = []
                
                if "donor" in ensemble_features and "acceptor" in ensemble_features:
                    suggestions.append("**设计氢键网络**: 同时包含供体和受体以增强结合亲和力")
                
                if "hydrophobe" in ensemble_features:
                    suggestions.append("**优化疏水区域**: 增强疏水相互作用以提高选择性")
                
                if "aromatic" in ensemble_features:
                    suggestions.append("**引入芳香环**: 增强π-π堆积作用")
                
                if suggestions:
                    for suggestion in suggestions:
                        st.success(suggestion)
                else:
                    st.info("基于当前药效团，建议综合考虑多种相互作用类型")

# 独立运行
if __name__ == "__main__":
    st.set_page_config(
        page_title="药效团生成器",
        page_icon="🎯",
        layout="wide"
    )
    render_pharmacophore_tab()