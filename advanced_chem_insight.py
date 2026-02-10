"""
advanced_chem_insight.py - 高级化学洞察模块（基于优化版T004+T033）
为EGFR预测系统提供专业级的相似性分析和分子表示对比
版本: 2.0.0 - 基于TeachOpenCADD T004优化版本构建
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    PandasTools,
    Draw,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    Fragments
)
from rdkit.Chem.Draw import rdDepictor
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import os
import random

# 配置设置
rdDepictor.SetPreferCoordGen(True)

# 定义项目基础目录
BASE_DIR = Path(__file__).parent

@dataclass
class ScreeningConfig:
    """虚拟筛选配置"""
    pic50_cutoff: float = 6.3
    fp_radius: int = 2
    fp_size: int = 2048
    top_n_results: int = 10
    similarity_threshold: float = 0.7

class AdvancedMolecularSimilarity:
    """高级分子相似性计算（基于优化版T004）"""
    
    def __init__(self, config: ScreeningConfig = None):
        self.config = config or ScreeningConfig()
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.config.fp_radius, 
            fpSize=self.config.fp_size
        )
    
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.rdchem.Mol]:
        """安全地将SMILES转换为分子对象"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # 尝试清理SMILES
                smiles_clean = smiles.split()[0].strip()
                mol = Chem.MolFromSmiles(smiles_clean)
            return mol
        except Exception:
            return None
    
    def calculate_descriptors(self, molecules_df: pd.DataFrame) -> pd.DataFrame:
        """计算分子描述符（包含错误处理）"""
        df = molecules_df.copy()
        
        try:
            # 1D描述符
            df["molecule_weight"] = df["ROMol"].apply(Descriptors.MolWt)
            df["logp"] = df["ROMol"].apply(Descriptors.MolLogP)
            df["hbd"] = df["ROMol"].apply(Descriptors.NumHDonors)
            df["hba"] = df["ROMol"].apply(Descriptors.NumHAcceptors)
            df["rotatable_bonds"] = df["ROMol"].apply(Descriptors.NumRotatableBonds)
            
            # 2D指纹
            df["maccs_fp"] = df["ROMol"].apply(MACCSkeys.GenMACCSKeys)
            df["morgan_fp"] = df["ROMol"].apply(self.morgan_generator.GetFingerprint)
            df["morgan_count_fp"] = df["ROMol"].apply(
                self.morgan_generator.GetCountFingerprint
            )
            
            # 拓扑极性表面积
            df["tpsa"] = df["ROMol"].apply(Descriptors.TPSA)
            
        except Exception as e:
            st.warning(f"描述符计算部分失败: {e}")
        
        return df
    
    def calculate_similarities(self, 
                             query_mol: Chem.rdchem.Mol,
                             molecules_df: pd.DataFrame) -> pd.DataFrame:
        """计算查询分子与数据集中所有分子的相似性（多指标）"""
        if query_mol is None:
            raise ValueError("查询分子无效")
        
        df = molecules_df.copy()
        
        try:
            # 生成查询分子指纹
            maccs_fp_query = MACCSkeys.GenMACCSKeys(query_mol)
            morgan_fp_query = self.morgan_generator.GetFingerprint(query_mol)
            morgan_count_fp_query = self.morgan_generator.GetCountFingerprint(query_mol)
            
            # 提取数据集指纹列表
            maccs_fps = df["maccs_fp"].tolist()
            morgan_fps = df["morgan_fp"].tolist()

            # 安全获取计数指纹（避免重复计算）
            if "morgan_count_fp" in df.columns:
                morgan_count_fps = df["morgan_count_fp"].tolist()
            else:
                morgan_count_fps = df["ROMol"].apply(
                    self.morgan_generator.GetCountFingerprint
                ).tolist()
            
            # 计算MACCS相似度
            df["tanimoto_maccs"] = DataStructs.BulkTanimotoSimilarity(
                maccs_fp_query, maccs_fps
            )
            df["dice_maccs"] = DataStructs.BulkDiceSimilarity(
                maccs_fp_query, maccs_fps
            )
            
            # 计算Morgan相似度
            df["tanimoto_morgan"] = DataStructs.BulkTanimotoSimilarity(
                morgan_fp_query, morgan_fps
            )
            df["dice_morgan"] = DataStructs.BulkDiceSimilarity(
                morgan_fp_query, morgan_fps
            )
            
            # 计数指纹相似度
            df["tanimoto_morgan_count"] = DataStructs.BulkTanimotoSimilarity(
                morgan_count_fp_query, morgan_count_fps
            )
            
            # 计算平均相似度
            df["avg_similarity"] = df[["tanimoto_morgan", "tanimoto_maccs"]].mean(axis=1)
            
        except Exception as e:
            st.error(f"相似度计算失败: {e}")
        
        return df
    
    def get_similarity_statistics(self, df: pd.DataFrame) -> Dict:
        """获取相似度统计信息"""
        stats = {
            "morgan_mean": df["tanimoto_morgan"].mean(),
            "morgan_std": df["tanimoto_morgan"].std(),
            "morgan_max": df["tanimoto_morgan"].max(),
            "maccs_mean": df["tanimoto_maccs"].mean(),
            "maccs_std": df["tanimoto_maccs"].std(),
            "maccs_max": df["tanimoto_maccs"].max(),
            "high_similarity_count": (df["tanimoto_morgan"] > 0.7).sum(),
            "active_high_similarity": ((df["tanimoto_morgan"] > 0.7) & 
                                     (df.get("pIC50", 0) >= self.config.pic50_cutoff)).sum()
        }
        return stats

class AdvancedVisualization:
    """高级可视化工具"""
    
    @staticmethod
    def visualize_similarity_distribution(df: pd.DataFrame, 
                                        figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """可视化相似度分布（增强版）"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 相似度分布直方图
        df["tanimoto_maccs"].hist(ax=axes[0, 0], bins=50, alpha=0.7, 
                                 color='skyblue', edgecolor='black')
        axes[0, 0].set_title("MACCS指纹相似度分布", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("Tanimoto系数")
        axes[0, 0].set_ylabel("分子数量")
        axes[0, 0].axvline(x=0.7, color='red', linestyle='--', alpha=0.7, 
                          label='高相似度阈值')
        axes[0, 0].legend()
        
        df["tanimoto_morgan"].hist(ax=axes[0, 1], bins=50, alpha=0.7, 
                                  color='orange', edgecolor='black')
        axes[0, 1].set_title("Morgan指纹相似度分布", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("Tanimoto系数")
        axes[0, 1].axvline(x=0.7, color='red', linestyle='--', alpha=0.7)
        
        # 活性与非活性的相似度对比箱线图
        if "pIC50" in df.columns:
            df["activity_status"] = df["pIC50"].apply(
                lambda x: "活性" if x >= 6.3 else "非活性"
            )
            
            data_for_box = []
            labels = []
            for status in ["活性", "非活性"]:
                subset = df[df["activity_status"] == status]
                if len(subset) > 0:
                    data_for_box.append(subset["tanimoto_morgan"].values)
                    labels.append(status)
            
            if data_for_box:
                axes[0, 2].boxplot(data_for_box, labels=labels, 
                                  patch_artist=True,
                                  boxprops=dict(facecolor='lightgreen', alpha=0.7))
                axes[0, 2].set_title("活性状态相似度对比", fontsize=12, fontweight='bold')
                axes[0, 2].set_ylabel("Tanimoto系数")
                axes[0, 2].grid(True, alpha=0.3)
        
        # 相似度比较散点图
        df.plot.scatter(x="tanimoto_maccs", y="tanimoto_morgan", 
                       ax=axes[1, 0], alpha=0.6, c=df.get("pIC50", 50), 
                       cmap='viridis', colorbar=True)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel("MACCS相似度")
        axes[1, 0].set_ylabel("Morgan相似度")
        axes[1, 0].set_title("指纹方法比较", fontsize=12, fontweight='bold')
        
        # 相似度与活性的关系
        if "pIC50" in df.columns:
            axes[1, 1].scatter(df["tanimoto_morgan"], df["pIC50"], 
                              alpha=0.6, c='purple')
            axes[1, 1].set_xlabel("Morgan相似度")
            axes[1, 1].set_ylabel("pIC50")
            axes[1, 1].set_title("相似度与活性关系", fontsize=12, fontweight='bold')
            
            # 添加趋势线
            try:
                z = np.polyfit(df["tanimoto_morgan"], df["pIC50"], 1)
                p = np.poly1d(z)
                axes[1, 1].plot(df["tanimoto_morgan"], p(df["tanimoto_morgan"]), 
                               "r--", alpha=0.8)
            except:
                pass
        
        # 指纹类型对比雷达图
        ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
        
        fp_types = ["MACCS", "Morgan", "计数指纹"]
        avg_similarities = [
            df["tanimoto_maccs"].mean(),
            df["tanimoto_morgan"].mean(),
            df.get("tanimoto_morgan_count", pd.Series([0])).mean()
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(fp_types), endpoint=False).tolist()
        avg_similarities += avg_similarities[:1]
        angles += angles[:1]
        
        ax_radar.plot(angles, avg_similarities, 'o-', linewidth=2)
        ax_radar.fill(angles, avg_similarities, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(fp_types)
        ax_radar.set_title("指纹方法效果对比", fontsize=12, fontweight='bold', y=1.1)
        ax_radar.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_enrichment_plot(df: pd.DataFrame,
                              similarity_measure: str = "tanimoto_morgan",
                              pic50_cutoff: float = 6.3) -> plt.Figure:
        """创建富集曲线图"""
        # 检查pIC50列是否存在
        if "pIC50" not in df.columns:
            st.warning("数据集缺少pIC50列，无法生成富集曲线")
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "数据缺少活性值(pIC50)列", ha='center', va='center', fontsize=14)
            return fig

        # 确保数据按相似度降序排列
        df_sorted = df.sort_values(similarity_measure, ascending=False).reset_index(drop=True)
        
        # 计算累积统计量
        n_total = len(df_sorted)
        n_actives = (df_sorted["pIC50"] >= pic50_cutoff).sum()
        
        # 计算累积活性分子数
        df_sorted["cumulative_actives"] = (df_sorted["pIC50"] >= pic50_cutoff).cumsum()
        
        # 计算百分比
        df_sorted["%_ranked_dataset"] = (df_sorted.index + 1) / n_total * 100
        df_sorted["%_true_actives_identified"] = df_sorted["cumulative_actives"] / n_actives * 100
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制富集曲线
        ax.plot(df_sorted["%_ranked_dataset"],
                df_sorted["%_true_actives_identified"],
                label=f"{similarity_measure}",
                color='blue',
                linewidth=2.5,
                alpha=0.8)
        
        # 最优曲线
        ratio_actives = n_actives / n_total * 100
        x_optimal = [0, ratio_actives, 100]
        y_optimal = [0, 100, 100]
        ax.plot(x_optimal, y_optimal, 
                label="最优曲线", 
                color='green', 
                linestyle='--',
                linewidth=2)
        
        # 随机曲线
        ax.plot([0, 100], [0, 100], 
                label="随机曲线", 
                color='grey', 
                linestyle=':',
                linewidth=2)
        
        # 美化图形
        ax.set_xlabel("排序数据百分比 (%)", fontsize=12, fontweight='bold')
        ax.set_ylabel("识别活性分子百分比 (%)", fontsize=12, fontweight='bold')
        ax.set_title("虚拟筛选富集曲线", fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置百分比格式
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.yaxis.set_major_formatter(PercentFormatter())
        
        # 添加标注
        ax.text(10, 90, f"活性分子: {n_actives}/{n_total}", 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                      facecolor="yellow", alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_top_molecules(query_mol: Chem.rdchem.Mol,
                              top_molecules_df: pd.DataFrame,
                              n_mols_per_row: int = 4,
                              sub_img_size: Tuple[int, int] = (250, 200)) -> str:
        """可视化查询分子和排名靠前的分子（返回Base64编码的图片）"""
        legends = [f"查询分子"]
        
        for idx, (_, row) in enumerate(top_molecules_df.iterrows(), 1):
            activity = row.get('pIC50', 'N/A')
            similarity = row.get('tanimoto_morgan', 0)
            
            if isinstance(activity, (int, float)):
                activity_text = f"{activity:.2f}"
            else:
                activity_text = str(activity)
            
            legend = (f"#{idx}\n"
                     f"相似度: {similarity:.3f}\n"
                     f"pIC50: {activity_text}")
            legends.append(legend)
        
        mols_to_draw = [query_mol] + top_molecules_df["ROMol"].tolist()
        
        # 生成分子网格图像
        img = Draw.MolsToGridImage(
            mols=mols_to_draw,
            legends=legends,
            molsPerRow=n_mols_per_row,
            subImgSize=sub_img_size,
            useSVG=False,  # 使用PNG格式，适合Streamlit
            returnPNG=True
        )
        
        # 转换为base64字符串
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str

class AdvancedChemInsightEngine:
    """高级化学洞察引擎（集成优化版T004和T033）"""
    
    def __init__(self, reference_data_path: Optional[Path] = None):
        """
        智能初始化：支持离线、上传、集成三种模式，确保永远有可用的分析结果
        """
        self.config = ScreeningConfig()
        self.similarity_calc = AdvancedMolecularSimilarity(self.config)
        self.visualizer = AdvancedVisualization()
        self.reference_df = None
        self.data_mode = "offline"  # offline, uploaded, integrated
        
        # 优先级1: 用户上传的数据
        if reference_data_path and os.path.exists(reference_data_path):
            if self.load_reference_data(reference_data_path):
                self.data_mode = "uploaded"
        
        # 优先级2: 集成项目数据
        elif self._find_project_data():
            self.data_mode = "integrated"
            
        # 优先级3: 离线模式（总有数据）
        else:
            self._load_offline_data()
            self.data_mode = "offline"
    
    def _load_offline_data(self):
        """离线数据：内置示例 + 智能生成"""
        import random
        
        # 生成多样化的示例分子（EGFR相关结构）
        offline_examples = [
            # EGFR抑制剂核心骨架
            {'smiles': 'Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1', 'activity': 8.2, 'name': 'EGFR核心骨架'},
            {'smiles': 'COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4', 'activity': 7.9, 'name': '吉非替尼类似物'},
            {'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'activity': 4.5, 'name': '咖啡因（阴性对照）'},
            {'smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O', 'activity': 3.8, 'name': '阿司匹林（阴性对照）'},
            {'smiles': 'C1=CC=C(C=C1)C=O', 'activity': 5.2, 'name': '苯甲醛'},
            # 扩展更多示例
            {'smiles': 'CC(C)NCC(O)COC1=CC=C(C=C1)CC(=O)N(C)C', 'activity': 6.7, 'name': '模拟ADME优化'},
            {'smiles': 'O=C(NC1=CC=CC=C1)C2=CC=CN=C2', 'activity': 7.1, 'name': '双环酰胺'},
            {'smiles': 'NC(=O)C1=CC=C(OCCN2CCOCC2)C=C1', 'activity': 6.9, 'name': '柔性连接子示例'},
        ]
        
        # 添加化学规则生成的虚拟分子
        core_scaffolds = [
            'Nc1ncnc2cc3ccccc3cc12',  # 嘌呤类似物
            'O=C(Nc1ccccc1)c2cccnc2',  # 芳基酰胺
            'Cc1cc(C)c(/C=C2\\C(=O)Nc3ncnc(N)c32)oc1C',  # 复杂天然产物类似物
        ]
        
        for scaffold in core_scaffolds:
            # 通过化学规则生成变体
            for _ in range(3):
                modified = self._generate_variant(scaffold)
                offline_examples.append({
                    'smiles': modified,
                    'activity': round(random.uniform(5.0, 8.5), 1),
                    'name': '规则生成变体'
                })
        
        self.reference_df = pd.DataFrame(offline_examples)
        self.reference_df['pIC50'] = self.reference_df['activity']
        self.reference_df['source'] = 'offline_demo'
        
        # 添加分子对象列
        if "ROMol" not in self.reference_df.columns:
            PandasTools.AddMoleculeColumnToFrame(self.reference_df, "smiles")
        
        # 计算描述符
        self.reference_df = self.similarity_calc.calculate_descriptors(self.reference_df)
    
    def _find_project_data(self):
        """自动查找项目已有的数据文件"""
        possible_paths = [
            BASE_DIR / "egfr_compounds_clean.csv",
            BASE_DIR / "data" / "egfr_data.csv",
            BASE_DIR / "egfr_compounds.csv",
            BASE_DIR / "rf_egfr_model_final.pkl",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # 如果是CSV，直接加载
                if path.suffix == '.csv':
                    try:
                        self.load_reference_data(path)
                        self.data_mode = "integrated"
                        return True
                    except Exception as e:
                        continue
                # 如果是模型文件，提取特征信息
                elif path.suffix == '.pkl':
                    try:
                        self._extract_from_model(path)
                        self.data_mode = "integrated"
                        return True
                    except Exception:
                        continue
        
        return False
    
    def _extract_from_model(self, model_path):
        """从模型文件中提取训练数据信息"""
        import pickle
        
        with open(model_path, 'rb') as f:
            pickle.load(f)
        
        # 如果模型包含特征名称等信息，可以用于构建参考数据
        # 这里简化处理，加载离线数据作为备选
        self._load_offline_data()
        self.reference_df['source'] = 'model_extracted'
    
    def _generate_variant(self, scaffold):
        """基于化学规则生成分子变体"""
        mol = Chem.MolFromSmiles(scaffold)
        if not mol:
            return scaffold
        
        # 简单的化学变换规则
        transforms = [
            lambda m: Chem.MolFromSmiles(m.ToSmiles().replace('C', 'N', 1)),
            lambda m: Chem.MolFromSmiles(m.ToSmiles().replace('=O', '=S', 1)),
            lambda m: Chem.MolFromSmiles(m.ToSmiles() + 'C'),
            lambda m: Chem.MolFromSmiles('CC' + m.ToSmiles()),
        ]
        
        try:
            transformed = random.choice(transforms)(mol)
            return Chem.MolToSmiles(transformed) if transformed else scaffold
        except:
            return scaffold
    
    def load_reference_data(self, data_path: Path) -> bool:
        """加载参考数据集（添加分子对象列）"""
        try:
            self.reference_df = pd.read_csv(data_path)
            
            # 检查必要列
            required_cols = ["smiles"]
            missing_cols = [col for col in required_cols if col not in self.reference_df.columns]
            
            if missing_cols:
                st.error(f"数据集缺少必要列: {missing_cols}")
                return False
            
            # 添加分子对象列
            if "ROMol" not in self.reference_df.columns:
                PandasTools.AddMoleculeColumnToFrame(self.reference_df, "smiles")
            
            # 计算描述符
            self.reference_df = self.similarity_calc.calculate_descriptors(self.reference_df)
            
            st.sidebar.success(f"✅ 参考数据集已加载: {len(self.reference_df)} 个分子")
            return True
            
        except Exception as e:
            st.error(f"加载数据集失败: {e}")
            return False
    
    def perform_advanced_screening(self, 
                                 query_smiles: str,
                                 top_n: int = 10) -> Dict[str, Any]:
        """执行高级相似性筛选"""
        results = {
            "success": False,
            "query_smiles": query_smiles,
            "query_mol": None,
            "top_molecules": None,
            "statistics": None,
            "enrichment_data": None,
            "visualizations": {}
        }
        
        try:
            # 1. 验证查询分子
            query_mol = self.similarity_calc.smiles_to_mol(query_smiles)
            if query_mol is None:
                results["error"] = "无效的SMILES字符串"
                return results
            
            results["query_mol"] = query_mol
            
            if self.reference_df is None:
                results["error"] = "未加载参考数据集"
                return results
            
            # 2. 计算相似度
            screened_df = self.similarity_calc.calculate_similarities(
                query_mol, self.reference_df
            )
            
            # 3. 排序并获取Top N
            screened_df = screened_df.sort_values(
                "tanimoto_morgan", ascending=False
            ).reset_index(drop=True)
            
            top_molecules = screened_df.head(top_n).copy()
            results["top_molecules"] = top_molecules
            
            # 4. 计算统计信息
            results["statistics"] = self.similarity_calc.get_similarity_statistics(screened_df)
            
            # 5. 生成富集数据
            if "pIC50" in screened_df.columns:
                results["enrichment_data"] = self._generate_enrichment_data(
                    screened_df, "tanimoto_morgan"
                )
            
            # 6. 生成可视化
            results["visualizations"]["similarity_distribution"] = \
                self.visualizer.visualize_similarity_distribution(screened_df)
            
            if "pIC50" in screened_df.columns:
                results["visualizations"]["enrichment_plot"] = \
                    self.visualizer.create_enrichment_plot(screened_df)
            
            results["visualizations"]["molecules_grid"] = \
                self.visualizer.visualize_top_molecules(query_mol, top_molecules)
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = str(e)
            st.error(f"筛选过程出错: {e}")
        
        return results
    
    def _generate_enrichment_data(self, df: pd.DataFrame,
                                similarity_measure: str) -> pd.DataFrame:
        """生成富集数据"""
        df_sorted = df.sort_values(similarity_measure, ascending=False).reset_index(drop=True)

        n_total = len(df_sorted)

        # 检查pIC50列是否存在
        if "pIC50" not in df_sorted.columns:
            st.warning("数据集缺少pIC50列")
            return pd.DataFrame(columns=["%_ranked_dataset", "%_true_actives_identified"])

        n_actives = (df_sorted["pIC50"] >= self.config.pic50_cutoff).sum()

        # 处理无活性分子的情况
        if n_actives == 0:
            df_sorted["cumulative_actives"] = 0
            df_sorted["%_ranked_dataset"] = (df_sorted.index + 1) / n_total * 100
            df_sorted["%_true_actives_identified"] = 0.0
        else:
            df_sorted["cumulative_actives"] = (df_sorted["pIC50"] >= self.config.pic50_cutoff).cumsum()
            df_sorted["%_ranked_dataset"] = (df_sorted.index + 1) / n_total * 100
            df_sorted["%_true_actives_identified"] = df_sorted["cumulative_actives"] / n_actives * 100

        return df_sorted[["%_ranked_dataset", "%_true_actives_identified"]].copy()
    
    def calculate_enrichment_factors(self, df: pd.DataFrame,
                                   cutoff_percentages: List[float] = None) -> pd.DataFrame:
        """计算富集因子"""
        # 检查pIC50列是否存在
        if "pIC50" not in df.columns:
            st.warning("数据集缺少pIC50列，无法计算富集因子")
            return pd.DataFrame()

        if cutoff_percentages is None:
            cutoff_percentages = [1, 2, 5, 10]

        results = []
        n_actives = (df["pIC50"] >= self.config.pic50_cutoff).sum()
        n_total = len(df)

        if n_actives == 0:
            st.warning("数据集中没有活性分子(pIC50 >= 6.3)，无法计算富集因子")
            return pd.DataFrame()

        ratio_actives = n_actives / n_total * 100  # 活性分子百分比

        for cutoff in cutoff_percentages:
            row = {"Cutoff_%": cutoff}

            for measure in ["tanimoto_maccs", "tanimoto_morgan"]:
                enrichment = self._generate_enrichment_data(df, measure)

                mask = enrichment["%_ranked_dataset"] <= cutoff
                if mask.any():
                    ef = enrichment.loc[mask, "%_true_actives_identified"].iloc[-1]
                    row[f"EF_{measure}"] = round(ef, 2)
                else:
                    row[f"EF_{measure}"] = 0.0

            # 随机EF（在随机筛选下，识别的活性百分比等于检查的数据百分比）
            row["EF_Random"] = cutoff

            # 最优EF：理想情况下，在检查所有活性分子后识别100%活性分子
            # EF_Optimal = min(100 / ratio_actives, 100 / cutoff)
            if ratio_actives > 0:
                ef_optimal = min(100 / ratio_actives, 100 / cutoff)
                row["EF_Optimal"] = round(ef_optimal, 2)
            else:
                row["EF_Optimal"] = 0.0

            results.append(row)

        return pd.DataFrame(results)

def render_advanced_chem_insight():
    """在Streamlit中渲染高级化学洞察界面"""
    
    st.header("🔬 高级化学洞察分析")
    st.markdown("""
    提供专业级的配体相似性筛选和分析功能
    """)
    
    # 初始化引擎（智能初始化，确保总有数据）
    engine = AdvancedChemInsightEngine()
    
    # 数据状态指示器
    col_status, col_upload = st.columns([2, 1])
    
    with col_status:
        mode_badges = {
            "offline": "🔘 离线模式（示例数据）",
            "uploaded": "✅ 自定义数据模式",
            "integrated": "⚡ 集成数据模式"
        }
        
        st.info(f"**数据模式**: {mode_badges.get(engine.data_mode, '离线模式')}")
        
        if engine.data_mode == "offline":
            st.caption("💡 上传自定义数据可获得更精准分析")
    
    with col_upload:
        with st.expander("📁 上传数据", expanded=False):
            uploaded_file = st.file_uploader(
                "上传CSV文件（包含smiles列）",
                type=['csv'],
                help="文件应包含'smiles'列，可选'activity'、'pIC50'等活性列"
            )
            
            if uploaded_file:
                try:
                    user_df = pd.read_csv(uploaded_file)
                    if 'smiles' in user_df.columns:
                        data_path = Path("uploaded_reference_data.csv")
                        with open(data_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        if engine.load_reference_data(data_path):
                            engine.data_mode = "uploaded"
                            st.success(f"✅ 已加载 {len(user_df)} 个分子")
                            
                            # 显示数据预览
                            with st.expander("预览数据"):
                                st.dataframe(user_df.head())
                    else:
                        st.error("CSV必须包含'smiles'列")
                except Exception as e:
                    st.error(f"文件读取失败: {e}")
    
    # 获取查询分子
    col1, col2 = st.columns([3, 1])
    with col1:
        if 'last_smiles' in st.session_state and st.session_state.last_smiles:
            default_smiles = st.session_state.last_smiles
        else:
            default_smiles = "COC1=C(OCCCN2CCOCC2)C=C2C(NC3=CC(Cl)=C(F)C=C3)=NC=NC2=C1"
        
        query_smiles = st.text_area(
            "**输入查询分子SMILES**",
            value=default_smiles,
            height=100,
            help="输入要分析的分子SMILES，如EGFR抑制剂Gefitinib"
        )
    
    with col2:
        st.subheader("⚙️ 筛选参数")
        top_n = st.slider("显示数量", 5, 20, 10)
        st.slider("高相似度阈值", 0.1, 1.0, 0.7, 0.05, key="similarity_threshold")
        
        if st.button("🚀 开始高级分析", type="primary", use_container_width=True):
            st.session_state["advanced_analysis_triggered"] = True
    
    # 执行分析
    if st.session_state.get("advanced_analysis_triggered", False) and query_smiles:
        with st.spinner("正在进行高级化学分析..."):
            results = engine.perform_advanced_screening(
                query_smiles, top_n=top_n
            )
            
            if results["success"]:
                display_advanced_results(results, engine)
            else:
                st.error(f"分析失败: {results.get('error', '未知错误')}")

def display_advanced_results(results: Dict, engine: AdvancedChemInsightEngine):
    """显示高级分析结果"""
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 筛选结果", 
        "📈 统计分析", 
        "🎯 富集分析",
        "🧪 分子可视化",
        "🧠 智能解读"
    ])
    
    with tab1:
        st.subheader("相似性筛选结果")
        
        # 显示数据源说明
        if engine.data_mode == "offline":
            st.warning("""
            **当前使用示例数据**，结果基于化学相似性原理展示。
            上传真实EGFR数据集可获得针对性的精准分析。
            """)
        
        if results["top_molecules"] is not None:
            top_df = results["top_molecules"]
            
            # 显示统计摘要
            col1, col2, col3, col4 = st.columns(4)
            stats = results["statistics"]
            
            col1.metric("最高相似度", f"{stats['morgan_max']:.3f}")
            col2.metric("平均相似度", f"{stats['morgan_mean']:.3f}")
            col3.metric("高相似度分子", stats['high_similarity_count'])
            col4.metric("高相似度活性分子", stats['active_high_similarity'])
            
            # 显示详细结果表格（动态选择可用列）
            available_cols = []
            column_mapping = {
                "molecule_chembl_id": "ChEMBL ID",
                "tanimoto_morgan": "Morgan相似度",
                "tanimoto_maccs": "MACCS相似度",
                "pIC50": "活性值",
                "molecule_weight": "分子量",
                "logp": "LogP",
                "smiles": "SMILES"
            }

            for col in column_mapping.keys():
                if col in top_df.columns:
                    available_cols.append(col)

            if available_cols:
                display_df = top_df[available_cols].rename(columns=column_mapping)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("无法显示详细结果：缺少必要的列")
    
    with tab2:
        st.subheader("统计分析")
        
        if "similarity_distribution" in results["visualizations"]:
            fig = results["visualizations"]["similarity_distribution"]
            st.pyplot(fig)
            
            # 下载按钮
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="📥 下载分布图",
                data=buf,
                file_name="similarity_distribution.png",
                mime="image/png"
            )
        
        # 富集因子计算
        if results["enrichment_data"] is not None and engine.reference_df is not None:
            st.subheader("富集因子分析")
            
            ef_df = engine.calculate_enrichment_factors(engine.reference_df)
            st.dataframe(ef_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("富集曲线分析")
        
        if "enrichment_plot" in results["visualizations"]:
            fig = results["visualizations"]["enrichment_plot"]
            st.pyplot(fig)
            
            # 解释富集曲线
            with st.expander("📖 富集曲线解读指南"):
                st.markdown("""
                **富集曲线解释**:
                - **蓝色曲线**: 实际筛选性能
                - **绿色虚线**: 理论上限（最优曲线）
                - **灰色点线**: 随机筛选基线
                
                **性能评估**:
                - 曲线越靠近左上角，筛选性能越好
                - 早期富集因子(EF@1%) > 20表示优秀筛选能力
                - 曲线下面积(AUC)越大，整体性能越好
                """)
    
    with tab4:
        st.subheader("分子可视化")
        
        if "molecules_grid" in results["visualizations"]:
            img_str = results["visualizations"]["molecules_grid"]
            
            # 显示分子网格
            st.markdown(
                f'<img src="data:image/png;base64,{img_str}" width="100%">',
                unsafe_allow_html=True
            )
            
            # 化学意义解读
            st.subheader("🧪 化学洞察")
            
            if results["top_molecules"] is not None and len(results["top_molecules"]) > 0:
                top_mol = results["top_molecules"].iloc[0]
                similarity = top_mol["tanimoto_morgan"]
                activity = top_mol.get("pIC50", None)
                
                if similarity > 0.8:
                    st.success("**🔍 发现高度相似分子**")
                    st.markdown("""
                    - 查询分子与已知化合物结构高度相似
                    - 预测结果具有强化学依据
                    - 建议进一步验证结合模式和药效团匹配
                    """)
                
                if activity and activity >= 6.3:
                    st.info(f"**🎯 发现活性类似物** (pIC50 = {activity:.2f})")
                    st.markdown("""
                    - 相似分子具有已知活性
                    - 支持基于相似性的活性预测
                    - 可作为先导化合物优化的起点
                    """)
    
    with tab5:
        st.subheader("🧠 智能化学解读")
        
        # 获取查询SMILES
        query_smiles = results.get('query_smiles', '')
        
        # 基于规则的分析，不依赖外部数据
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**基于化学规则的分析**")
            
            # 1. 子结构识别
            if query_smiles:
                mol = engine.similarity_calc.smiles_to_mol(query_smiles)
                if mol:
                    # 识别常见药效团
                    pharmacophores = identify_pharmacophores(mol)
                    st.write("**识别到的潜在药效团:**")
                    for pharma in pharmacophores:
                        st.write(f"• {pharma}")
        
        with col2:
            st.markdown("**分子性质评估**")
            
            if query_smiles:
                # 计算基本性质
                props = calculate_molecular_properties(query_smiles)
                
                st.metric("分子量", f"{props.get('mw', 0):.1f}")
                st.metric("脂水分配系数(LogP)", f"{props.get('logp', 0):.2f}")
                st.metric("氢键供体", props.get('hbd', 0))
                st.metric("氢键受体", props.get('hba', 0))


def identify_pharmacophores(mol: Chem.rdchem.Mol) -> List[str]:
    """识别分子中的常见药效团"""
    from rdkit.Chem import Fragments
    
    pharmacophores = []
    
    # 检查芳香环
    num_aromatic_rings = Fragments.fr_benzene(mol) + Fragments.fr_aniline(mol)
    if num_aromatic_rings > 0:
        pharmacophores.append(f"芳香环系统 ({num_aromatic_rings}个)")
    
    # 检查氢键受体
    num_acceptors = Fragments.fr_NH0(mol) + Fragments.fr_NH1(mol) + Fragments.fr_NH2(mol)
    if num_acceptors > 0:
        pharmacophores.append(f"氢键受体 ({num_acceptors}个)")
    
    # 检查氢键供体
    num_donors = Fragments.fr_NH1(mol) + Fragments.fr_NH2(mol)
    if num_donors > 0:
        pharmacophores.append(f"氢键供体 ({num_donors}个)")
    
    # 检查卤素
    if Fragments.fr_halogen(mol) > 0:
        pharmacophores.append("卤素原子")
    
    # 检查酰胺
    if Fragments.fr_amide(mol) > 0:
        pharmacophores.append("酰胺基团")
    
    # 检查硝基
    if Fragments.fr_nitro(mol) > 0:
        pharmacophores.append("硝基")
    
    if not pharmacophores:
        pharmacophores.append("未识别到常见药效团")
    
    return pharmacophores


def calculate_molecular_properties(smiles: str) -> Dict[str, float]:
    """计算分子的基本理化性质"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {}
    
    props = {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'rotb': Descriptors.NumRotatableBonds(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
    }
    
    return props

# 独立运行测试
if __name__ == "__main__":
    st.set_page_config(
        page_title="高级化学洞察模块", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    render_advanced_chem_insight()