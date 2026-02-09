"""
详细诊断RF模型加载失败的原因
"""
import os
import sys
import traceback

print("=" * 70)
print("🔍 RF模型加载问题详细诊断")
print("=" * 70)

# 1. 检查当前工作目录
print(f"\n1️⃣ 当前工作目录: {os.getcwd()}")

# 2. 检查脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"2️⃣ 脚本所在目录: {current_dir}")

# 3. 检查numpy版本
import numpy as np
print(f"3️⃣ NumPy版本: {np.__version__}")

# 4. 检查joblib版本
import joblib
print(f"4️⃣ Joblib版本: {joblib.__version__}")

# 5. 检查模型文件是否存在
print(f"\n5️⃣ 检查模型文件...")
model_path = os.path.join(current_dir, "rf_egfr_model_final.pkl")
compatible_model_path = os.path.join(current_dir, "rf_egfr_model_compatible.pkl")

print(f"   原始模型路径: {model_path}")
print(f"   原始模型存在: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"   原始模型大小: {size_mb:.2f} MB")

print(f"\n   兼容模型路径: {compatible_model_path}")
print(f"   兼容模型存在: {os.path.exists(compatible_model_path)}")
if os.path.exists(compatible_model_path):
    size_mb = os.path.getsize(compatible_model_path) / (1024*1024)
    print(f"   兼容模型大小: {size_mb:.2f} MB")

# 6. 检查feature_names.json
feature_path = os.path.join(current_dir, "feature_names.json")
print(f"\n6️⃣ 检查特征文件...")
print(f"   特征文件路径: {feature_path}")
print(f"   特征文件存在: {os.path.exists(feature_path)}")
if os.path.exists(feature_path):
    with open(feature_path, 'r', encoding='utf-8') as f:
        import json
        features = json.load(f)
    print(f"   特征数量: {len(features)}")
    print(f"   前5个特征: {features[:5]}")

# 7. 尝试加载原始模型
print(f"\n7️⃣ 尝试加载原始模型...")
try:
    model = joblib.load(model_path)
    print(f"   ✅ 原始模型加载成功")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   模型属性: {dir(model)[:10]}")

    if hasattr(model, 'n_features_in_'):
        print(f"   期望特征数: {model.n_features_in_}")

    if hasattr(model, 'feature_importances_'):
        print(f"   特征重要性: {len(model.feature_importances_)} 个")

except Exception as e:
    print(f"   ❌ 原始模型加载失败: {e}")
    print(f"\n   详细错误:")
    traceback.print_exc()

# 8. 尝试加载兼容模型
if os.path.exists(compatible_model_path):
    print(f"\n8️⃣ 尝试加载兼容模型...")
    try:
        model = joblib.load(compatible_model_path)
        print(f"   ✅ 兼容模型加载成功")
        print(f"   模型类型: {type(model).__name__}")

        if hasattr(model, 'n_features_in_'):
            print(f"   期望特征数: {model.n_features_in_}")

    except Exception as e:
        print(f"   ❌ 兼容模型加载失败: {e}")
        traceback.print_exc()

# 9. 尝试导入RealEGFRPredictor
print(f"\n9️⃣ 尝试导入RealEGFRPredictor类...")
try:
    from real_predictor import RealEGFRPredictor
    print(f"   ✅ RealEGFRPredictor导入成功")

    # 初始化
    print(f"\n🔟 初始化RealEGFRPredictor...")
    predictor = RealEGFRPredictor()

    if predictor.model is None:
        print(f"   ❌ 模型为None - 加载失败")
    else:
        print(f"   ✅ 模型加载成功")
        print(f"   特征数量: {len(predictor.feature_names)}")

        # 尝试预测
        print(f"\n🧪 尝试进行预测...")
        test_smiles = "CN1CCN(CC1)C2=NC3=C(N2)NC(NC3=O)C"
        result = predictor.predict(test_smiles)

        if "error" in result:
            print(f"   ❌ 预测失败: {result['error']}")
        else:
            print(f"   ✅ 预测成功!")
            print(f"   预测结果: {'活性' if result['prediction']==1 else '非活性'}")
            print(f"   活性概率: {result['probability_active']:.3f}")

except Exception as e:
    print(f"   ❌ RealEGFRPredictor初始化失败: {e}")
    print(f"\n   详细错误:")
    traceback.print_exc()

# 10. 总结
print(f"\n" + "=" * 70)
print("📊 诊断总结")
print("=" * 70)

issues = []

if not os.path.exists(model_path):
    issues.append("原始模型文件不存在")

if not os.path.exists(feature_path):
    issues.append("特征文件不存在")

try:
    import joblib
    joblib.load(model_path)
except Exception as e:
    issues.append(f"模型加载失败: {str(e)[:50]}")

if issues:
    print(f"\n❌ 发现 {len(issues)} 个问题:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print(f"\n✅ 所有检查通过！")
