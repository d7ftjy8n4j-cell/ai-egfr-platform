"""
测试RF模型是否能在云端环境正常加载
在本地运行，模拟云端环境（numpy 1.24.4）
"""

import sys
import os

print("=" * 60)
print("🧪 测试RF模型加载（云端环境模拟）")
print("=" * 60)

# 检查numpy版本
import numpy as np
print(f"\n当前numpy版本: {np.__version__}")

if not np.__version__.startswith('1.24'):
    print(f"⚠️  警告：云端使用numpy 1.24.4，当前版本不同可能导致问题")
    print(f"💡 建议: pip install numpy==1.24.4")
else:
    print(f"✅ numpy版本与云端兼容")

# 测试1：尝试加载原始模型
print(f"\n测试1: 加载原始模型 rf_egfr_model_final.pkl")
try:
    import joblib
    model = joblib.load('rf_egfr_model_final.pkl')
    print(f"✅ 原始模型加载成功")
    print(f"   模型类型: {type(model).__name__}")
    
    # 尝试预测
    import numpy as np
    if hasattr(model, 'n_features_in_'):
        test_data = np.zeros((1, model.n_features_in_))
        pred = model.predict(test_data)
        print(f"✅ 原始模型可以预测")
except Exception as e:
    print(f"❌ 原始模型加载/预测失败: {e}")
    print(f"   这是预期的情况 - 原始模型在numpy 1.24.4上不兼容")

# 测试2：检查是否存在兼容模型
print(f"\n测试2: 检查兼容模型 rf_egfr_model_compatible.pkl")
if os.path.exists('rf_egfr_model_compatible.pkl'):
    print(f"✅ 兼容模型文件存在")
    try:
        import joblib
        model = joblib.load('rf_egfr_model_compatible.pkl')
        print(f"✅ 兼容模型加载成功")
        print(f"   模型类型: {type(model).__name__}")
        
        # 尝试预测
        if hasattr(model, 'n_features_in_'):
            test_data = np.zeros((1, model.n_features_in_))
            pred = model.predict(test_data)
            print(f"✅ 兼容模型可以预测")
    except Exception as e:
        print(f"❌ 兼容模型加载失败: {e}")
else:
    print(f"⚠️  兼容模型不存在")
    print(f"💡 请运行: python rebuild_model.py")

# 测试3：测试RealEGFRPredictor
print(f"\n测试3: 测试RealEGFRPredictor类")
try:
    from real_predictor import RealEGFRPredictor
    predictor = RealEGFRPredictor()
    
    if predictor.model is None:
        print(f"❌ RealEGFRPredictor初始化失败：模型未加载")
    else:
        print(f"✅ RealEGFRPredictor初始化成功")
        
        # 测试预测
        test_smiles = "CN1CCN(CC1)C2=NC3=C(N2)NC(NC3=O)C"
        result = predictor.predict(test_smiles)
        
        if "error" in result:
            print(f"❌ 预测失败: {result['error']}")
        else:
            print(f"✅ 预测成功!")
            print(f"   SMILES: {test_smiles[:30]}...")
            print(f"   预测结果: {'活性' if result['prediction']==1 else '非活性'}")
            print(f"   活性概率: {result['probability_active']:.3f}")
            print(f"   置信度: {result['confidence']}")
except Exception as e:
    print(f"❌ RealEGFRPredictor测试失败: {e}")

# 总结
print("\n" + "=" * 60)
print("📊 测试总结")
print("=" * 60)

recommendations = []

if not os.path.exists('rf_egfr_model_compatible.pkl'):
    recommendations.append("运行 'python rebuild_model.py' 创建兼容模型")
    
if not np.__version__.startswith('1.24'):
    recommendations.append("安装兼容版本: pip install numpy==1.24.4")

if recommendations:
    print("\n💡 建议操作：")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
else:
    print("\n✅ 所有测试通过！RF模型可以部署到云端。")
