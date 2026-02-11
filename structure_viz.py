"""
structure_viz.py
适配 Streamlit 的蛋白质结构可视化工具 (移除 PLIP 分析依赖)
基于 TeachOpenCADD T017 重构，使用 py3Dmol 替代 NGLView
"""

import py3Dmol
import requests
from stmol import showmol
import streamlit as st

class StructureVisualizer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.pdb_data = None
        self.pdb_id = None

    def load_from_pdb_id(self, pdb_id):
        """从 RCSB 下载 PDB 数据"""
        self.pdb_id = pdb_id
        try:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url)
            if response.status_code == 200:
                self.pdb_data = response.text
                return True
            else:
                return False
        except Exception as e:
            st.error(f"下载出错: {e}")
            return False

    def load_from_file(self, uploaded_file):
        """加载上传的文件内容"""
        self.pdb_id = uploaded_file.name
        # py3Dmol 需要字符串格式的 PDB 数据
        self.pdb_data = uploaded_file.getvalue().decode("utf-8")
        return True

    def render_view(self, style='cartoon', color_scheme='spectrum', show_ligand=True, show_surface=False, surface_opacity=0.5):
        """
        生成 py3Dmol 视图对象
        
        Parameters:
        - style: protein style ('cartoon', 'line', 'cross', 'stick', 'sphere')
        - color_scheme: protein color ('spectrum', 'chain', 'residue', etc.)
        - show_ligand: boolean, whether to show heteroatoms
        - show_surface: boolean, whether to calculate and show VDW surface
        """
        if not self.pdb_data:
            return None

        # 创建视图
        view = py3Dmol.view(width=self.width, height=self.height)
        view.addModel(self.pdb_data, "pdb")

        # 1. 设置蛋白质样式
        if style == 'cartoon':
            view.setStyle({'cartoon': {'color': color_scheme}})
        elif style == 'line':
            view.setStyle({'line': {'color': color_scheme}})
        elif style == 'stick':
            view.setStyle({'stick': {'color': color_scheme}})
        elif style == 'sphere':
            view.setStyle({'sphere': {'color': color_scheme}})
        
        # 2. 显式添加配体 (Heteroatoms)
        if show_ligand:
            # 选择所有非水杂原子
            # py3Dmol 选择器: hetflag=true 且 resn != HOH
            view.addStyle(
                {'hetflag': True, 'not': {'resn': ['HOH', 'WAT']}},
                {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.25}}
            )
            # 可选：给配体加个标签（由于不知道具体残基名，这里暂不加，以免标签满天飞）

        # 3. 添加表面 (计算量较大，作为可选项)
        if show_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': surface_opacity, 'color': 'white'}, {'protein': True})

        # 4. 视角设置
        view.zoomTo()
        
        return view
