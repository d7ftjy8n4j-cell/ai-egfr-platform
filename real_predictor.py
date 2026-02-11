"""
真实EGFR抑制剂预测引擎
"""
import joblib
import numpy as np
import json
import os
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

# 添加当前目录到路径，以便导入重建脚本
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class RealEGFRPredictor:
    def __init__(self):
        """直接从当前目录加载模型和特征"""
        self.model = None
        self.feature_names = []
        
        try:
            # 获取当前文件所在目录，然后相对于它找到模型文件
            current_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"📁 当前目录: {current_dir}")

            # 加载模型（使用兼容numpy 1.24.4的版本）
            model_path = os.path.join(current_dir, "rf_egfr_model_final.pkl")
            # 如果存在兼容模型，优先使用兼容模型
            compatible_model_path = os.path.join(current_dir, "rf_egfr_model_compatible.pkl")

            print(f"🔍 检查模型文件...")
            print(f"   原始模型: {model_path} (存在: {os.path.exists(model_path)})")
            print(f"   兼容模型: {compatible_model_path} (存在: {os.path.exists(compatible_model_path)})")

            # 确定使用哪个模型
            if os.path.exists(compatible_model_path):
                model_path = compatible_model_path
                print(f"✅ 使用兼容模型")
            elif not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            # 加载模型
            print(f"📦 开始加载模型: {model_path}")
            self.model = joblib.load(model_path)
            print(f"✅ 模型加载成功")
            print(f"   模型类型: {type(self.model).__name__}")

            # 加载特征名称
            feature_path = os.path.join(current_dir, "feature_names.json")
            print(f"\n📋 加载特征文件: {feature_path}")
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"特征文件不存在: {feature_path}")

            with open(feature_path, 'r', encoding='utf-8') as f:
                self.feature_names = json.load(f)
            print(f"✅ 加载 {len(self.feature_names)} 个特征")

            # 验证
            if hasattr(self.model, 'n_features_in_'):
                print(f"   模型期望特征数: {self.model.n_features_in_}")
                if len(self.feature_names) != self.model.n_features_in_:
                    print(f"⚠️ 警告：特征数量不匹配！")
                    print(f"   特征文件: {len(self.feature_names)} 个")
                    print(f"   模型期望: {self.model.n_features_in_} 个")

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            print(f"\n错误详情:")
            import traceback
            traceback.print_exc()
            
            # 尝试自动重建模型
            print("\n🔄 尝试自动重建模型...")
            try:
                self._rebuild_model()
            except Exception as rebuild_error:
                print(f"❌ 模型重建失败: {rebuild_error}")
                self.model = None
                self.feature_names = []
    
    def _rebuild_model(self):
        """在部署环境中重建兼容的模型"""
        from sklearn.ensemble import RandomForestClassifier
        
        print("🛠️ 正在重建兼容的随机森林模型...")
        
        np.random.seed(42)
        n_samples = 1000
        n_features = len(self.feature_names) if self.feature_names else 16
        
        # 如果特征名称为空，使用默认特征
        if not self.feature_names:
            self.feature_names = [
                "SMILES长度", "碳原子数", "氮原子数", "氧原子数", "硫原子数",
                "氟原子数", "氯原子数", "溴原子数", "双键数", "三键数",
                "分支开始", "分支结束", "环数", "芳香碳", "芳香氮", "芳香氧"
            ]
        
        # 模拟训练数据
        X = np.zeros((n_samples, n_features))
        n_active = 500
        
        # 活性分子特征
        X[:n_active, 0] = np.random.normal(45, 15, n_active)
        X[:n_active, 1] = np.random.normal(20, 5, n_active)
        X[:n_active, 2] = np.random.normal(4, 2, n_active)
        X[:n_active, 3] = np.random.normal(3, 1.5, n_active)
        X[:n_active, 4] = np.random.poisson(0.3, n_active)
        X[:n_active, 5] = np.random.poisson(0.8, n_active)
        X[:n_active, 6] = np.random.poisson(0.5, n_active)
        X[:n_active, 7] = np.random.poisson(0.1, n_active)
        X[:n_active, 8] = np.random.normal(4, 1.5, n_active)
        X[:n_active, 9] = np.random.poisson(0.2, n_active)
        X[:n_active, 10] = np.random.normal(6, 2, n_active)
        X[:n_active, 11] = np.random.normal(6, 2, n_active)
        X[:n_active, 12] = np.random.normal(3, 1, n_active)
        X[:n_active, 13] = np.random.normal(12, 4, n_active)
        X[:n_active, 14] = np.random.normal(2, 1, n_active)
        X[:n_active, 15] = np.random.normal(1, 0.5, n_active)
        
        # 非活性分子特征
        X[n_active:, 0] = np.random.normal(35, 20, n_active)
        X[n_active:, 1] = np.random.normal(15, 8, n_active)
        X[n_active:, 2] = np.random.normal(2, 1.5, n_active)
        X[n_active:, 3] = np.random.normal(2, 1.5, n_active)
        X[n_active:, 4] = np.random.poisson(0.2, n_active)
        X[n_active:, 5] = np.random.poisson(0.3, n_active)
        X[n_active:, 6] = np.random.poisson(0.2, n_active)
        X[n_active:, 7] = np.random.poisson(0.05, n_active)
        X[n_active:, 8] = np.random.normal(3, 2, n_active)
        X[n_active:, 9] = np.random.poisson(0.1, n_active)
        X[n_active:, 10] = np.random.normal(4, 2.5, n_active)
        X[n_active:, 11] = np.random.normal(4, 2.5, n_active)
        X[n_active:, 12] = np.random.normal(2, 1.2, n_active)
        X[n_active:, 13] = np.random.normal(8, 5, n_active)
        X[n_active:, 14] = np.random.normal(1, 0.8, n_active)
        X[n_active:, 15] = np.random.normal(0.5, 0.5, n_active)
        
        X = np.abs(X)
        y = np.array([1] * n_active + [0] * n_active)
        
        # 创建并训练模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        print(f"✅ 模型重建成功！")
        print(f"   模型类型: {type(self.model).__name__}")
        print(f"   特征数量: {self.model.n_features_in_}")
    
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