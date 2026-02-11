"""
protein_ligand_streamlit.py - 蛋白质-配体相互作用分析模块
基于TeachOpenCADD的T016+T017教程，适配Streamlit界面
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import requests
import warnings
warnings.filterwarnings("ignore")

# 尝试导入所需的库
try:
    # PLIP用于蛋白质-配体相互作用分析
    from plip.structure.preparation import PDBComplex
    from plip.exchange.report import BindingSiteReport

    # 3D可视化
    import py3Dmol

    PLIP_AVAILABLE = True
    st.sidebar.success("✅ PLIP分析模块就绪")
except ImportError as e:
    PLIP_AVAILABLE = False
    st.sidebar.warning(f"⚠️ PLIP分析模块不可用: {e}")

class StreamlitProteinLigandAnalyzer:
    """
    适配Streamlit的蛋白质-配体相互作用分析器
    """
    
    def __init__(self):
        """初始化分析器"""
        # 存储临时文件
        self.temp_dir = Path(tempfile.gettempdir()) / "protein_ligand_analysis"
        self.temp_dir.mkdir(exist_ok=True)
        
        # 存储相互作用数据
        self.interactions_by_site = None
        self.selected_site = None
        self.pdb_file_path = None
        
        # 相互作用类型颜色映射
        self.color_map = {
            "hydrophobic": "#FFD700",  # 金色
            "hbond": "#4169E1",        # 蓝色
            "waterbridge": "#32CD32",  # 绿色
            "saltbridge": "#FF4500",   # 橙色
            "pistacking": "#8A2BE2",   # 紫色
            "pication": "#00CED1",     # 青色
            "halogen": "#FF1493",      # 粉色
            "metal": "#A9A9A9",        # 灰色
        }
    
    def download_pdb(self, pdb_id):
        """
        从RCSB PDB下载PDB文件
        
        Parameters
        ----------
        pdb_id : str
            PDB ID（如'3poz', '1aaq'）
            
        Returns
        -------
        str : PDB文件路径
        """
        pdb_id = pdb_id.lower().strip()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # 保存到临时文件
            pdb_file = self.temp_dir / f"{pdb_id}.pdb"
            with open(pdb_file, 'w') as f:
                f.write(response.text)
            
            st.success(f"✅ 成功下载PDB: {pdb_id.upper()}")
            return str(pdb_file)
            
        except Exception as e:
            st.error(f"❌ 下载PDB失败: {pdb_id.upper()}")
            st.error(f"错误详情: {str(e)}")
            return None
    
    def analyze_interactions(self, pdb_file_path):
        """
        使用PLIP分析蛋白质-配体相互作用
        
        Parameters
        ----------
        pdb_file_path : str
            PDB文件路径
            
        Returns
        -------
        dict : 包含所有结合位点相互作用的字典
        """
        if not PLIP_AVAILABLE:
            st.error("PLIP库未安装，无法分析相互作用")
            return {}
        
        try:
            # 创建PLIP复合物对象
            protlig = PDBComplex()
            protlig.load_pdb(pdb_file_path)
            
            # 寻找配体并分析相互作用
            for ligand in protlig.ligands:
                protlig.characterize_complex(ligand)
            
            sites = {}
            # 遍历所有结合位点
            for key, site in sorted(protlig.interaction_sets.items()):
                binding_site = BindingSiteReport(site)
                
                # 要提取的相互作用类型
                keys = (
                    "hydrophobic", "hbond", "waterbridge", "saltbridge",
                    "pistacking", "pication", "halogen", "metal"
                )
                
                # 提取相互作用信息
                interactions = {
                    k: [getattr(binding_site, k + "_features")] + 
                        getattr(binding_site, k + "_info")
                    for k in keys
                }
                sites[key] = interactions
            
            self.interactions_by_site = sites
            self.pdb_file_path = pdb_file_path
            
            st.success(f"✅ 分析完成！发现 {len(sites)} 个结合位点")
            return sites
            
        except Exception as e:
            st.error(f"❌ PLIP分析失败: {str(e)}")
            return {}
    
    def create_interaction_dataframe(self, site_index=0, interaction_type="all"):
        """
        为特定相互作用类型创建DataFrame
        
        Parameters
        ----------
        site_index : int
            结合位点索引
        interaction_type : str
            相互作用类型，'all'表示所有类型
            
        Returns
        -------
        pd.DataFrame : 包含相互作用细节的DataFrame
        """
        if not self.interactions_by_site:
            return pd.DataFrame()
        
        # 获取选定的结合位点
        sites = list(self.interactions_by_site.keys())
        if site_index >= len(sites):
            st.warning(f"位点索引 {site_index} 超出范围，可用位点: {len(sites)}")
            return pd.DataFrame()
        
        site_key = sites[site_index]
        site_interactions = self.interactions_by_site[site_key]
        
        if interaction_type == "all":
            # 合并所有相互作用类型
            all_dfs = []
            for int_type, int_list in site_interactions.items():
                if len(int_list) > 1:  # 有数据
                    df = pd.DataFrame.from_records(
                        int_list[1:],
                        columns=int_list[0]
                    )
                    df['interaction_type'] = int_type
                    all_dfs.append(df)
            
            if all_dfs:
                return pd.concat(all_dfs, ignore_index=True)
            else:
                return pd.DataFrame()
        else:
            # 特定相互作用类型
            if interaction_type not in site_interactions:
                return pd.DataFrame()
            
            int_list = site_interactions[interaction_type]
            if len(int_list) <= 1:
                return pd.DataFrame()
            
            df = pd.DataFrame.from_records(
                int_list[1:],
                columns=int_list[0]
            )
            df['interaction_type'] = interaction_type
            return df
    
    def visualize_structure_3d(self, pdb_id=None, highlight_residues=None):
        """
        使用py3Dmol在Streamlit中可视化3D结构
        
        Parameters
        ----------
        pdb_id : str
            PDB ID（用于在线加载）
        highlight_residues : list
            要高亮的残基编号列表
            
        Returns
        -------
        stmol组件
        """
        if pdb_id and not self.pdb_file_path:
            # 在线加载PDB
            pdb_data = f"https://files.rcsb.org/view/{pdb_id}.pdb"
        elif self.pdb_file_path:
            # 从文件加载
            with open(self.pdb_file_path, 'r') as f:
                pdb_data = f.read()
        else:
            st.error("没有可用的PDB数据")
            return None
        
        try:
            # 创建3D视图
            view = py3Dmol.view(width=700, height=500)
            
            if isinstance(pdb_data, str) and pdb_data.startswith('http'):
                # 在线PDB
                view.addModel(requests.get(pdb_data).text, 'pdb')
            else:
                # 本地PDB数据
                view.addModel(pdb_data, 'pdb')
            
            # 设置可视化样式
            view.setStyle({'model': -1}, {
                'cartoon': {'color': 'spectrum'},
                'stick': {'radius': 0.15}
            })
            
            # 高亮配体
            view.addStyle({'resn': []}, {
                'stick': {'colorscheme': 'orangeCarbon', 'radius': 0.3}
            })
            
            # 如果有高亮残基
            if highlight_residues:
                for res in highlight_residues:
                    view.addStyle({'resi': res}, {
                        'stick': {'colorscheme': 'redCarbon', 'radius': 0.3},
                        'cartoon': {'color': 'red'}
                    })
            
            # 设置背景和视角
            view.setBackgroundColor('white')
            view.zoomTo()

            # 将 view 对象转换为 HTML，并用 Streamlit 组件渲染
            html_code = view._repr_html_()
            components.html(html_code, height=500, width=700)

        except Exception as e:
            st.error(f"3D可视化失败: {str(e)}")
            return None
    
    def generate_interaction_summary(self):
        """
        生成相互作用总结报告
        
        Returns
        -------
        dict : 总结统计信息
        """
        if not self.interactions_by_site:
            return {}
        
        summary = {
            "total_sites": len(self.interactions_by_site),
            "site_details": {},
            "total_interactions": 0
        }
        
        for site_key, site_data in self.interactions_by_site.items():
            site_summary = {}
            total_site_interactions = 0
            
            for int_type, int_list in site_data.items():
                count = len(int_list) - 1  # 减去标题行
                if count > 0:
                    site_summary[int_type] = count
                    total_site_interactions += count
            
            summary["site_details"][site_key] = {
                "interactions": site_summary,
                "total": total_site_interactions
            }
            summary["total_interactions"] += total_site_interactions
        
        return summary
    
    def plot_interaction_chart(self, summary_data):
        """
        绘制相互作用类型的统计图表
        
        Parameters
        ----------
        summary_data : dict
            总结数据
        """
        if not summary_data or "site_details" not in summary_data:
            return
        
        import matplotlib.pyplot as plt
        
        # 获取第一个位点的数据
        first_site_key = list(summary_data["site_details"].keys())[0]
        site_interactions = summary_data["site_details"][first_site_key]["interactions"]
        
        if not site_interactions:
            return
        
        # 创建条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        types = list(site_interactions.keys())
        counts = list(site_interactions.values())
        colors = [self.color_map.get(t, "#808080") for t in types]
        
        bars = ax.barh(types, counts, color=colors)
        ax.set_xlabel("相互作用数量", fontsize=12)
        ax.set_title(f"结合位点 {first_site_key} 的相互作用类型分布", fontsize=14, pad=20)
        
        # 在条形上添加数值标签
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f"{count}", va='center', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)

def render_protein_ligand_tab():
    """
    渲染蛋白质-配体相互作用分析的Streamlit标签页
    """
    st.header("🔬 蛋白质-配体相互作用分析")
    
    # 创建分析器实例
    if 'pl_analyzer' not in st.session_state:
        st.session_state.pl_analyzer = StreamlitProteinLigandAnalyzer()
    
    analyzer = st.session_state.pl_analyzer
    
    # 侧边栏配置
    with st.sidebar:
        st.subheader("⚙️ 分析设置")
        
        input_method = st.radio(
            "选择输入方式:",
            ["使用PDB ID", "上传PDB文件"]
        )
        
        if input_method == "使用PDB ID":
            col1, col2 = st.columns([3, 1])
            with col1:
                pdb_id = st.text_input(
                    "PDB ID:",
                    value="3poz",
                    help="输入PDB ID，如：3poz (EGFR激酶), 1aaq (HIV蛋白酶)"
                )
            with col2:
                st.markdown("")  # 占位符
                st.markdown("")  # 占位符
                if st.button("🔍 获取", use_container_width=True):
                    with st.spinner("正在下载PDB文件..."):
                        analyzer.pdb_file_path = analyzer.download_pdb(pdb_id)
                        
        else:  # 上传文件
            uploaded_file = st.file_uploader(
                "上传PDB文件:",
                type=['pdb'],
                help="上传本地PDB文件进行分析"
            )
            if uploaded_file:
                # 保存上传的文件
                temp_file = analyzer.temp_dir / uploaded_file.name
                with open(temp_file, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                analyzer.pdb_file_path = str(temp_file)
                st.success(f"✅ 已上传: {uploaded_file.name}")
        
        st.divider()
        
        # 分析控制
        st.subheader("🔬 分析控制")
        
        if st.button("🚀 开始相互作用分析", use_container_width=True, type="primary"):
            if analyzer.pdb_file_path:
                with st.spinner("正在分析蛋白质-配体相互作用..."):
                    analyzer.analyze_interactions(analyzer.pdb_file_path)
            else:
                st.warning("请先提供PDB文件")
        
        st.divider()
        
        # 可视化选项
        st.subheader("👁️ 可视化选项")
        
        show_structure = st.checkbox("显示3D结构", value=True)
        show_interactions = st.checkbox("显示相互作用表", value=True)
        show_summary = st.checkbox("显示统计摘要", value=True)
    
    # 主内容区
    if analyzer.pdb_file_path:
        # 如果有相互作用数据，显示分析结果
        if analyzer.interactions_by_site:
            # 1. 3D结构可视化
            if show_structure:
                st.subheader("🎨 3D结构可视化")
                st.info("""
                **颜色说明**: 
                - 蛋白质: 彩色卡通表示 (二级结构)
                - 配体: 橙色球棍模型
                - 相互作用残基: 红色高亮
                """)
                
                # 从相互作用数据中提取要高亮的残基
                if analyzer.interactions_by_site:
                    first_site = list(analyzer.interactions_by_site.keys())[0]
                    first_site_data = analyzer.interactions_by_site[first_site]
                    
                    # 提取所有相互作用的残基编号
                    highlight_residues = set()
                    for int_type, int_list in first_site_data.items():
                        if len(int_list) > 1:
                            df = pd.DataFrame.from_records(
                                int_list[1:],
                                columns=int_list[0]
                            )
                            if 'RESNR' in df.columns:
                                residues = df['RESNR'].unique()
                                highlight_residues.update(residues)
                    
                    # 显示3D结构
                    analyzer.visualize_structure_3d(highlight_residues=list(highlight_residues))
            
            # 2. 相互作用数据表
            if show_interactions:
                st.subheader("📊 相互作用数据")
                
                # 选择结合位点
                site_options = list(analyzer.interactions_by_site.keys())
                selected_site_idx = st.selectbox(
                    "选择结合位点:",
                    range(len(site_options)),
                    format_func=lambda i: f"位点 {i+1}: {site_options[i]}"
                )
                
                # 选择相互作用类型
                int_types = ["all"] + list(analyzer.color_map.keys())
                selected_int_type = st.selectbox(
                    "选择相互作用类型:",
                    int_types,
                    format_func=lambda x: "所有类型" if x == "all" else x
                )
                
                # 显示数据表
                df = analyzer.create_interaction_dataframe(
                    site_index=selected_site_idx,
                    interaction_type=selected_int_type
                )
                
                if not df.empty:
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # 导出选项
                    csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 导出CSV",
                        data=csv_data,
                        file_name=f"protein_ligand_interactions.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("该位点/相互作用类型没有发现相互作用")
            
            # 3. 统计摘要
            if show_summary:
                st.subheader("📈 相互作用统计")
                
                summary = analyzer.generate_interaction_summary()
                
                if summary:
                    # 显示关键指标
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("结合位点", summary["total_sites"])
                    with col2:
                        st.metric("总相互作用数", summary["total_interactions"])
                    with col3:
                        first_site = list(summary["site_details"].keys())[0]
                        st.metric("主位点相互作用", 
                                 summary["site_details"][first_site]["total"])
                    
                    # 绘制图表
                    analyzer.plot_interaction_chart(summary)
                    
                    # 详细总结
                    with st.expander("📋 详细总结报告"):
                        for site_key, site_info in summary["site_details"].items():
                            st.markdown(f"**结合位点: {site_key}**")
                            for int_type, count in site_info["interactions"].items():
                                st.write(f"- {int_type}: {count} 个相互作用")
                            st.write(f"**总计**: {site_info['total']} 个相互作用")
                            st.divider()
        
        else:
            # 提示开始分析
            st.info("👆 点击侧边栏的『开始相互作用分析』按钮，分析蛋白质-配体相互作用")
    
    else:
        # 初始状态
        st.info("""
        ## 🧬 蛋白质-配体相互作用分析
        
        **功能说明**:
        1. **输入PDB结构**: 通过PDB ID或上传PDB文件提供蛋白质-配体复合物结构
        2. **PLIP分析**: 自动识别结合位点，分析8种相互作用类型
        3. **3D可视化**: 交互式查看蛋白质-配体复合物结构
        4. **数据导出**: 导出详细的相互作用数据
        
        **支持的相互作用类型**:
        - 疏水相互作用 (hydrophobic)
        - 氢键 (hbond)
        - 水桥 (waterbridge)
        - 盐桥 (saltbridge)
        - π-π堆积 (pistacking)
        - π-阳离子 (pication)
        - 卤键 (halogen)
        - 金属配位 (metal)
        
        **示例PDB ID**:
        - `3poz`: EGFR激酶与抑制剂复合物
        - `1aaq`: HIV蛋白酶与抑制剂复合物
        - `1pdb`: 胰蛋白酶与抑制剂复合物
        """)
        
        # 快速示例
        st.subheader("🚀 快速开始示例")
        
        example_cols = st.columns(4)
        examples = [
            ("3poz", "EGFR激酶"),
            ("1aaq", "HIV蛋白酶"),
            ("1pdb", "胰蛋白酶"),
            ("1fkg", "FK506结合蛋白")
        ]
        
        for idx, (pdb_id, desc) in enumerate(examples):
            with example_cols[idx]:
                if st.button(f"🔬 {pdb_id}", use_container_width=True, key=f"ex_{pdb_id}"):
                    with st.spinner(f"正在获取{desc}结构..."):
                        analyzer.pdb_file_path = analyzer.download_pdb(pdb_id)
                    st.rerun()

# 如果直接运行此模块，显示独立的界面
if __name__ == "__main__":
    st.set_page_config(
        page_title="蛋白质-配体相互作用分析",
        page_icon="🔬",
        layout="wide"
    )
    render_protein_ligand_tab()