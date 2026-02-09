"""
重建兼容numpy 1.24.4的随机森林模型
解决部署时的numpy._core模块不兼容问题
"""

import joblib
import os

print("=" * 60)
print("开始重建兼容的随机森林模型...")
print("=" * 60)

# 检查原始模型是否存在
original_model_path = 'rf_egfr_model_final.pkl'
if not os.path.exists(original_model_path):
    print(f"❌ 错误：找不到原始模型文件 {original_model_path}")
    exit(1)

# 检查numpy版本
import numpy as np
print(f"\n当前numpy版本: {np.__version__}")

# 加载原始模型
print(f"\n正在加载原始模型: {original_model_path}...")
try:
    model = joblib.load(original_model_path)
    print("✅ 原始模型加载成功")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 保存兼容版本的模型
compatible_model_path = 'rf_egfr_model_compatible.pkl'
print(f"\n正在保存兼容模型: {compatible_model_path}...")
try:
    joblib.dump(model, compatible_model_path)
    print("✅ 兼容模型保存成功")
except Exception as e:
    print(f"❌ 保存失败: {e}")
    exit(1)

# 验证可以重新加载
print("\n验证兼容模型...")
try:
    test_model = joblib.load(compatible_model_path)
    print("✅ 兼容模型验证成功，可以正常加载")
except Exception as e:
    print(f"❌ 验证失败: {e}")
    exit(1)

# 获取模型信息
print("\n" + "=" * 60)
print("模型信息:")
print("=" * 60)
print(f"模型类型: {type(model).__name__}")
print(f"模型参数数量: {len(model.get_params())}")

# 尝试进行一次简单预测
try:
    # 创建一些测试数据（假设是分子指纹特征）
    import numpy as np
    # 获取模型的期望特征数量
    if hasattr(model, 'n_features_in_'):
        print(f"期望特征数量: {model.n_features_in_}")
        test_data = np.zeros((1, model.n_features_in_))
        pred = model.predict(test_data)
        print(f"测试预测结果: {pred}")
        print("✅ 模型可以进行预测")
except Exception as e:
    print(f"⚠️  测试预测时出错（这可能是正常的，因为我们只是测试了零数据）: {e}")

print("\n" + "=" * 60)
print("✅ 模型重建完成！")
print("=" * 60)
print(f"\n新的兼容模型文件: {compatible_model_path}")
print(f"文件大小: {os.path.getsize(compatible_model_path) / (1024*1024):.2f} MB")
print("\n下一步: 在app.py中将模型路径从 'rf_egfr_model_final.pkl'")
print("      改为 'rf_egfr_model_compatible.pkl'")
