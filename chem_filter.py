"""
chem_filter.py
药物筛选模块：ADME规则 (Lipinski) 和 不良子结构 (PAINS/Brenk)
基于 TeachOpenCADD T002 & T003
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from rdkit import Chem
from rdkit.Chem import Descriptors, PandasTools
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import streamlit as st
import io

# ==================== ADME 筛选 ====================

class ADMEFilter:
    """ADME筛选器，基于Lipinski's Rule of Five"""
    
    RO5_THRESHOLDS = {
        "molecular_weight": 500,  # Da
        "n_hba": 10,             # 氢键受体数
        "n_hbd": 5,              # 氢键供体数
        "logp": 5                # LogP值
    }

    @staticmethod
    def calculate_ro5_properties(smiles: str) -> pd.Series:
        """计算单分子的Ro5属性"""
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return pd.Series([None]*5, index=["MW", "HBA", "HBD", "LogP", "Pass_Ro5"])
            
            mw = Descriptors.ExactMolWt(molecule)
            hba = Descriptors.NumHAcceptors(molecule)
            hbd = Descriptors.NumHDonors(molecule)
            logp = Descriptors.MolLogP(molecule)
            
            conditions = [
                mw <= ADMEFilter.RO5_THRESHOLDS["molecular_weight"],
                hba <= ADMEFilter.RO5_THRESHOLDS["n_hba"],
                hbd <= ADMEFilter.RO5_THRESHOLDS["n_hbd"],
                logp <= ADMEFilter.RO5_THRESHOLDS["logp"]
            ]
            
            # 允许违反一项规则
            ro5_fulfilled = sum(conditions) >= 3
            
            return pd.Series(
                [mw, hba, hbd, logp, ro5_fulfilled],
                index=["MW", "HBA", "HBD", "LogP", "Pass_Ro5"]
            )
        except Exception as e:
            return pd.Series([None]*5, index=["MW", "HBA", "HBD", "LogP", "Pass_Ro5"])

    @staticmethod
    def plot_radar_chart(stats_df: pd.DataFrame, title: str = ""):
        """绘制雷达图 (适配 Streamlit，返回 fig)"""
        if stats_df.empty:
            return None

        def _scale_by_thresholds(stats, thresholds, scaled_threshold):
            # 简单的归一化逻辑
            normalized = []
            # MW
            normalized.append(stats.loc["MW"] / thresholds["molecular_weight"] * scaled_threshold)
            # HBA
            normalized.append(stats.loc["HBA"] / thresholds["n_hba"] * scaled_threshold)
            # HBD
            normalized.append(stats.loc["HBD"] / thresholds["n_hbd"] * scaled_threshold)
            # LogP
            normalized.append(stats.loc["LogP"] / thresholds["logp"] * scaled_threshold)
            
            return pd.Series(normalized, index=["MW", "HBA", "HBD", "LogP"])

        thresholds = ADMEFilter.RO5_THRESHOLDS
        scaled_threshold = 5
        properties_labels = ["MW (Da)", "# HBA", "# HBD", "LogP"]

        # 准备数据
        means = stats_df["mean"]
        # stnds = stats_df["std"] # 简化展示，暂不画标准差范围以免图形混乱

        # 归一化
        y = _scale_by_thresholds(means, thresholds, scaled_threshold)
        values = y.tolist()
        values += values[:1] # 闭合

        angles = [n / float(len(properties_labels)) * 2 * math.pi for n in range(len(properties_labels))]
        angles += angles[:1] # 闭合

        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        # 绘制阈值区域
        ax.fill(angles, [scaled_threshold] * len(angles), "cornflowerblue", alpha=0.2, label="Rule of Five")
        
        # 绘制平均值
        ax.plot(angles, values, "o-", linewidth=2, color="orange", label="Dataset Mean")
        ax.fill(angles, values, "orange", alpha=0.1)

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(properties_labels)

        # 设置Y轴
        ax.set_rlabel_position(0)
        plt.yticks([2.5, 5.0, 7.5], ["50%", "Limit", "150%"], color="grey", size=8)
        plt.ylim(0, 8)

        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.title(title, y=1.08)
        
        return fig

# ==================== 子结构筛选 ====================

class SubstructureFilter:
    """不良子结构筛选器 (PAINS & Brenk)"""
    
    def __init__(self):
        # 初始化 PAINS
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_catalog = FilterCatalog(params)
        
        # 初始化 Brenk (简化版，直接嵌入常用的 SMARTS)
        # 注意：完整的 Brenk 需要加载外部文件，这里使用代码中内置的常用列表
        self.brenk_data = {
            "Michael-acceptor": "[CX3]=[CX3].[CX3]=[CX3]",
            "Oxygen-nitrogen-single-bond": "O-N",
            "Nitro-group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
            "Aliphatic-long-chain": "CCCCCC",
            "Halogenated-ring": "c1ccc(Cl)cc1"
        }
        self.brenk_mols = {name: Chem.MolFromSmarts(s) for name, s in self.brenk_data.items()}

    def check_single_molecule(self, smiles: str):
        """检查单个分子"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}
            
        result = {
            "PAINS_found": False,
            "PAINS_names": [],
            "Brenk_found": False,
            "Brenk_names": []
        }
        
        # PAINS 检查
        if self.pains_catalog.HasMatch(mol):
            matches = self.pains_catalog.GetMatches(mol)
            for entry in matches:
                result["PAINS_names"].append(entry.GetDescription())
            result["PAINS_found"] = True
            
        # Brenk 检查
        for name, pattern in self.brenk_mols.items():
            if pattern and mol.HasSubstructMatch(pattern):
                result["Brenk_names"].append(name)
                result["Brenk_found"] = True
                
        return result

    def filter_dataframe(self, df: pd.DataFrame, smiles_col="smiles"):
        """批量筛选 DataFrame"""
        clean_indices = []
        pains_indices = []
        brenk_indices = []
        
        results = []
        
        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row[smiles_col])
            if mol is None:
                continue
                
            pains = False
            brenk = False
            issues = []
            
            # Check PAINS
            if self.pains_catalog.HasMatch(mol):
                pains = True
                matches = self.pains_catalog.GetMatches(mol)
                issues.append(f"PAINS: {[m.GetDescription() for m in matches]}")
                pains_indices.append(idx)
                
            # Check Brenk
            for name, pattern in self.brenk_mols.items():
                if pattern and mol.HasSubstructMatch(pattern):
                    brenk = True
                    issues.append(f"Brenk: {name}")
                    if idx not in brenk_indices: brenk_indices.append(idx)
            
            if not pains and not brenk:
                clean_indices.append(idx)
            
            results.append("; ".join(issues) if issues else "Pass")
            
        df["Filter_Status"] = results
        return df.loc[clean_indices], df, len(pains_indices), len(brenk_indices)
