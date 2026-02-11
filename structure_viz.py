"""
structure_viz.py
适配 Streamlit 的蛋白质结构可视化工具
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
        self.pdb_id = pdb_id
        try:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url)
            if response.status_code == 200:
                self.pdb_data = response.text
                return True
            return False
        except:
            return False

    def load_from_file(self, uploaded_file):
        self.pdb_id = uploaded_file.name
        self.pdb_data = uploaded_file.getvalue().decode("utf-8")
        return True

    def render_view(self, style='cartoon', color_scheme='spectrum', show_ligand=True, show_surface=False, surface_opacity=0.5):
        if not self.pdb_data:
            return None
        
        view = py3Dmol.view(width=self.width, height=self.height)
        view.addModel(self.pdb_data, "pdb")
        
        # 样式
        styles = {
            'cartoon': {'cartoon': {'color': color_scheme}},
            'line': {'line': {'color': color_scheme}},
            'stick': {'stick': {'color': color_scheme}},
            'sphere': {'sphere': {'color': color_scheme}}
        }
        view.setStyle(styles.get(style, styles['cartoon']))
        
        # 配体
        if show_ligand:
            view.addStyle(
                {'hetflag': True, 'not': {'resn': ['HOH', 'WAT']}},
                {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.25}}
            )
            
        # 表面
        if show_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': surface_opacity, 'color': 'white'}, {'protein': True})
            
        view.zoomTo()
        return view
