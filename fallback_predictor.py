"""
备用预测器 - 当模型文件无法加载时使用
基于SMILES字符串分析的简单EGFR抑制剂预测
完全不依赖RDKit
"""
import numpy as np

class FallbackEGFRPredictor:
    """
    基于SMILES字符串分析的EGFR抑制剂预测器
    当主模型加载失败时作为备用
    """
    
    def __init__(self):
        print("⚠️ 使用备用预测器（基于SMILES分析）")
        self.feature_names = [
            "SMILES长度", "碳原子数", "氮原子数", "氧原子数", "硫原子数",
            "氟原子数", "氯原子数", "溴原子数", "双键数", "三键数",
            "分支开始", "分支结束", "环数", "芳香碳", "芳香氮", "芳香氧"
        ]
    
    def _parse_smiles_simple(self, smiles):
        """
        简单解析SMILES字符串，不依赖RDKit
        """
        # 基本统计
        length = len(smiles)
        
        # 原子计数
        c_count = smiles.count('C') - smiles.count('Cl')  # 粗略估计
        n_count = smiles.count('N')
        o_count = smiles.count('O')
        s_count = smiles.count('S')
        f_count = smiles.count('F')
        cl_count = smiles.count('Cl')
        br_count = smiles.count('Br')
        
        # 键计数
        double_bonds = smiles.count('=')
        triple_bonds = smiles.count('#')
        
        # 分支
        branch_start = smiles.count('(')
        branch_end = smiles.count(')')
        
        # 环
        rings = sum(1 for c in smiles if c.isdigit())
        
        # 芳香性（小写c, n, o, s表示芳香原子）
        aromatic_c = smiles.count('c')
        aromatic_n = smiles.count('n')
        aromatic_o = smiles.count('o')
        aromatic_s = smiles.count('s')
        
        return {
            'length': length,
            'C': max(c_count, 0),
            'N': n_count,
            'O': o_count,
            'S': s_count,
            'F': f_count,
            'Cl': cl_count,
            'Br': br_count,
            'double_bonds': double_bonds,
            'triple_bonds': triple_bonds,
            'branch_start': branch_start,
            'branch_end': branch_end,
            'rings': rings // 2,  # 每个环有两个数字标记
            'aromatic_c': aromatic_c,
            'aromatic_n': aromatic_n,
            'aromatic_o': aromatic_o,
            'aromatic_s': aromatic_s
        }
    
    def predict(self, smiles):
        """基于SMILES字符串分析进行预测"""
        try:
            # 验证SMILES基本格式
            if not smiles or len(smiles) < 2:
                return {"error": "无效的SMILES字符串"}
            
            # 解析SMILES
            features = self._parse_smiles_simple(smiles)
            
            # 基于EGFR抑制剂的典型特征进行评分
            score = 0.0
            
            # 1. SMILES长度（典型EGFR抑制剂在30-70个字符）
            if 30 <= features['length'] <= 70:
                score += 0.15
            elif 20 <= features['length'] <= 100:
                score += 0.08
            
            # 2. 碳原子数（典型范围15-30）
            if 15 <= features['C'] <= 30:
                score += 0.15
            elif 10 <= features['C'] <= 40:
                score += 0.08
            
            # 3. 氮原子数（典型范围2-6，EGFR抑制剂通常有多个氮）
            if 2 <= features['N'] <= 6:
                score += 0.20
            elif 1 <= features['N'] <= 8:
                score += 0.10
            
            # 4. 氧原子数（典型范围1-5）
            if 1 <= features['O'] <= 5:
                score += 0.10
            
            # 5. 卤素（氟、氯、溴，EGFR抑制剂常有）
            halogen_score = features['F'] + features['Cl'] + features['Br']
            if 1 <= halogen_score <= 3:
                score += 0.15
            
            # 6. 芳香环（通过芳香原子数估算，EGFR抑制剂通常有2-4个芳香环）
            aromatic_atoms = features['aromatic_c'] + features['aromatic_n'] + features['aromatic_o']
            if 6 <= aromatic_atoms <= 18:  # 约2-3个芳香环
                score += 0.20
            elif 3 <= aromatic_atoms <= 24:
                score += 0.10
            
            # 7. 环数量
            if 2 <= features['rings'] <= 5:
                score += 0.05
            
            # 添加随机性
            np.random.seed(hash(smiles) % 2**32)
            noise = np.random.normal(0, 0.08)
            probability = np.clip(score + noise, 0.05, 0.95)
            
            # 预测类别
            pred_class = int(probability > 0.5)
            
            # 特征值列表
            feature_values = [
                features['length'],
                features['C'],
                features['N'],
                features['O'],
                features['S'],
                features['F'],
                features['Cl'],
                features['Br'],
                features['double_bonds'],
                features['triple_bonds'],
                features['branch_start'],
                features['branch_end'],
                features['rings'],
                features['aromatic_c'],
                features['aromatic_n'],
                features['aromatic_o'] + features['aromatic_s']
            ]
            
            # 特征重要性解释
            explanation = {
                "top_features": ["氮原子数", "碳原子数", "芳香原子数", "卤素原子数", "SMILES长度"],
                "top_importance": [0.20, 0.15, 0.20, 0.15, 0.15],
                "values": {
                    "氮原子数": features['N'],
                    "碳原子数": features['C'],
                    "芳香原子数": aromatic_atoms,
                    "卤素原子数": halogen_score,
                    "SMILES长度": features['length']
                }
            }
            
            return {
                "success": True,
                "smiles": smiles,
                "prediction": pred_class,
                "probability_active": float(probability),
                "confidence": "高" if abs(probability-0.5) > 0.3 else "中" if abs(probability-0.5) > 0.15 else "低",
                "explanation": explanation,
                "features_used": self.feature_names,
                "feature_values": feature_values,
                "note": "使用备用预测器（基于SMILES分析，无RDKit）"
            }
            
        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}


# 如果RDKit可用，创建更精确的备用预测器
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    
    class FallbackEGFRPredictorWithRDKit(FallbackEGFRPredictor):
        """带RDKit支持的增强版备用预测器"""
        
        def predict(self, smiles):
            """使用RDKit进行更精确的预测"""
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                # RDKit解析失败，回退到简单分析
                return super().predict(smiles)
            
            try:
                # 计算分子描述符
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
                n_rotatable = Descriptors.NumRotatableBonds(mol)
                
                # 评分
                score = 0.0
                
                if 300 <= mw <= 600:
                    score += 0.30
                elif 250 <= mw <= 700:
                    score += 0.15
                
                if 2 <= logp <= 5:
                    score += 0.20
                elif 1 <= logp <= 6:
                    score += 0.10
                
                if 1 <= hbd <= 4:
                    score += 0.15
                
                if 3 <= hba <= 10:
                    score += 0.15
                
                if 2 <= n_aromatic_rings <= 4:
                    score += 0.20
                
                # 添加随机性
                np.random.seed(hash(smiles) % 2**32)
                noise = np.random.normal(0, 0.05)
                probability = np.clip(score + noise, 0.05, 0.95)
                
                pred_class = int(probability > 0.5)
                
                # 构建特征值
                atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                feature_values = [
                    len(smiles),
                    atoms.count('C'),
                    atoms.count('N'),
                    atoms.count('O'),
                    atoms.count('S'),
                    atoms.count('F'),
                    atoms.count('Cl'),
                    atoms.count('Br'),
                    0, 0, 0, 0, 0, 0, 0, 0
                ]
                
                explanation = {
                    "top_features": ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "NumAromaticRings"],
                    "top_importance": [0.30, 0.20, 0.15, 0.15, 0.20],
                    "values": {
                        "MolWt": mw,
                        "MolLogP": logp,
                        "NumHDonors": hbd,
                        "NumHAcceptors": hba,
                        "NumAromaticRings": n_aromatic_rings
                    }
                }
                
                return {
                    "success": True,
                    "smiles": smiles,
                    "prediction": pred_class,
                    "probability_active": float(probability),
                    "confidence": "高" if abs(probability-0.5) > 0.3 else "中" if abs(probability-0.5) > 0.15 else "低",
                    "explanation": explanation,
                    "features_used": self.feature_names,
                    "feature_values": feature_values,
                    "note": "使用备用预测器（基于RDKit规则）"
                }
                
            except Exception as e:
                # RDKit计算失败，回退到简单分析
                return super().predict(smiles)
    
    # 如果RDKit可用，使用增强版
    FallbackEGFRPredictor = FallbackEGFRPredictorWithRDKit
    
except ImportError:
    # RDKit不可用，使用基础版
    pass
