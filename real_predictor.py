"""
真实EGFR抑制剂预测引擎 - 桌面版本
"""
import joblib
import numpy as np
import json
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

class RealEGFRPredictor:
    def __init__(self):
        """直接从桌面加载模型和特征"""
        try:
            # 桌面路径
            desktop = os.path.expanduser("~/Desktop")
            
            # 加载模型
            model_path = os.path.join(desktop, "rf_egfr_model_final.pkl")
            self.model = joblib.load(model_path)
            print(f"✅ 模型加载成功: {model_path}")
            
            # 加载特征名称
            feature_path = os.path.join(desktop, "feature_names.json")
            with open(feature_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
            print(f"✅ 加载 {len(self.feature_names)} 个特征")
            
            # 验证
            if hasattr(self.model, 'n_features_in_'):
                if len(self.feature_names) != self.model.n_features_in_:
                    print(f"⚠️ 特征数量不匹配！请检查feature_names.json")
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.feature_names = []
    
    def smiles_to_features(self, smiles):
        """将SMILES转换为模型所需的特征向量"""
        if not self.model:
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"❌ 无效的SMILES: {smiles}")
            return None
        
        features = []
        for feat_name in self.feature_names:
            try:
                # 获取RDKit计算函数
                func = getattr(Descriptors, feat_name)
                value = func(mol)
                features.append(float(value) if not pd.isna(value) else 0.0)
            except:
                features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, smiles):
        """主预测函数"""
        if not self.model:
            return {"error": "模型未加载"}
        
        features = self.smiles_to_features(smiles)
        if features is None:
            return {"error": "SMILES解析失败"}
        
        try:
            # 预测
            proba = self.model.predict_proba(features)[0]
            pred_class = int(proba[1] > 0.5)
            
            # 特征重要性解释
            explanation = None
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                # 取最重要的5个特征
                top5_idx = importances.argsort()[-5:][::-1]
                explanation = {
                    "top_features": [self.feature_names[i] for i in top5_idx],
                    "top_importance": [importances[i] for i in top5_idx],
                    "values": {self.feature_names[i]: features[0][i] for i in top5_idx}
                }
            
            return {
                "success": True,
                "smiles": smiles,
                "prediction": pred_class,  # 0=非活性, 1=活性
                "probability_active": float(proba[1]),
                "confidence": "高" if abs(proba[1]-0.5) > 0.3 else "中",
                "explanation": explanation,
                "features_used": self.feature_names,
                "feature_values": features[0].tolist()
            }
            
        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}

# 测试代码
if __name__ == "__main__":
    # 测试
    print("🧪 测试真实预测器...")
    
    # 初始化
    predictor = RealEGFRPredictor()
    
    # 测试吉非替尼
    gefitinib = "COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
    result = predictor.predict(gefitinib)
    
    if "error" in result:
        print(f"❌ 测试失败: {result['error']}")
    else:
        print(f"✅ 测试成功!")
        print(f"SMILES: {result['smiles'][:50]}...")
        print(f"预测: {'活性' if result['prediction']==1 else '非活性'}")
        print(f"活性概率: {result['probability_active']:.3f}")
        
        if result['explanation']:
            print("\n🔬 最重要的5个特征:")
            for i, (feat, imp) in enumerate(zip(result['explanation']['top_features'], 
                                               result['explanation']['importance']), 1):
                val = result['explanation']['values'][feat]
                print(f"  {i}. {feat}: 值={val:.2f}, 重要性={imp:.4f}")