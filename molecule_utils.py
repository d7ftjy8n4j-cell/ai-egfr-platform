"""
molecule_utils.py - 安全的分子处理工具
防止因SMILES解析失败导致的程序崩溃
"""

import logging
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Crippen, Lipinski

class SafeMolecule:
    """安全的分子处理类"""
    
    @staticmethod
    def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
        """
        安全地从SMILES创建分子对象
        返回: 分子对象或None
        """
        if not smiles or not isinstance(smiles, str):
            logging.warning(f"无效的SMILES输入: {smiles}")
            return None
        
        try:
            # 清理SMILES
            smiles_clean = smiles.strip()
            
            # 尝试解析
            mol = Chem.MolFromSmiles(smiles_clean)
            
            if mol is None:
                # 尝试修复常见问题
                smiles_clean = SafeMolecule._fix_smiles(smiles_clean)
                mol = Chem.MolFromSmiles(smiles_clean, sanitize=False)
                
                if mol is None:
                    logging.warning(f"无法解析SMILES: {smiles[:50]}...")
                    return None
            
            # 尝试进行价态检查
            try:
                Chem.SanitizeMol(mol)
            except:
                # 如果sanitize失败，使用非sanitize版本
                logging.warning(f"分子sanitize失败，使用非sanitize版本: {smiles[:50]}...")
            
            return mol
            
        except Exception as e:
            logging.error(f"SMILES解析异常: {smiles[:50]}... - {e}")
            return None
    
    @staticmethod
    def _fix_smiles(smiles: str) -> str:
        """尝试修复常见的SMILES问题"""
        fixes = [
            # 修复常见的格式问题
            (r'(\d+)([A-Z][a-z]?)', r'[\1\2]'),  # 同位素表示
            (r'([+-]{1,2})(?!\])', r'[\1]'),     # 电荷表示
            (r'\[H\]', 'H'),                     # 氢原子
            (r'\s+', ''),                        # 移除空格
            (r'\.{2,}', '.'),                    # 多个点
        ]
        
        fixed = smiles
        for pattern, replacement in fixes:
            import re
            fixed = re.sub(pattern, replacement, fixed)
        
        return fixed
    
    @staticmethod
    def safe_calculate_properties(mol: Chem.Mol) -> dict:
        """
        安全地计算分子性质
        返回: 包含性质的字典
        """
        if mol is None:
            return {
                'error': '分子对象为None',
                'mw': 0,
                'logp': 0,
                'hbd': 0,
                'hba': 0,
                'rotatable_bonds': 0,
                'aromatic_rings': 0,
                'tpsa': 0,
                'heavy_atoms': 0,
                'formal_charge': 0
            }
        
        try:
            # 使用安全的计算方式
            properties = {}
            
            # 分子量
            try:
                properties['mw'] = Descriptors.ExactMolWt(mol)
            except:
                properties['mw'] = 0
            
            # LogP
            try:
                properties['logp'] = Crippen.MolLogP(mol)
            except:
                properties['logp'] = 0
            
            # 氢键供体/受体
            try:
                properties['hbd'] = Lipinski.NumHDonors(mol)
                properties['hba'] = Lipinski.NumHAcceptors(mol)
            except:
                properties['hbd'] = 0
                properties['hba'] = 0
            
            # 其他性质
            try:
                properties['rotatable_bonds'] = Lipinski.NumRotatableBonds(mol)
            except:
                properties['rotatable_bonds'] = 0
            
            try:
                properties['aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
            except:
                properties['aromatic_rings'] = 0
            
            try:
                properties['tpsa'] = rdMolDescriptors.CalcTPSA(mol)
            except:
                properties['tpsa'] = 0
            
            try:
                properties['heavy_atoms'] = mol.GetNumHeavyAtoms()
            except:
                properties['heavy_atoms'] = 0
            
            try:
                properties['formal_charge'] = Chem.GetFormalCharge(mol)
            except:
                properties['formal_charge'] = 0
            
            return properties
            
        except Exception as e:
            logging.error(f"性质计算失败: {e}")
            return {
                'error': str(e),
                'mw': 0,
                'logp': 0,
                'hbd': 0,
                'hba': 0,
                'rotatable_bonds': 0,
                'aromatic_rings': 0,
                'tpsa': 0,
                'heavy_atoms': 0,
                'formal_charge': 0
            }
    
    @staticmethod
    def validate_smiles(smiles: str) -> tuple[bool, Optional[Chem.Mol], str]:
        """
        验证SMILES并返回解析结果
        返回: (是否有效, 分子对象, 错误信息)
        """
        mol = SafeMolecule.safe_mol_from_smiles(smiles)
        
        if mol is None:
            return False, None, "无法解析SMILES字符串"
        
        # 检查分子是否为空
        if mol.GetNumAtoms() == 0:
            return False, None, "分子不包含任何原子"
        
        return True, mol, "有效"