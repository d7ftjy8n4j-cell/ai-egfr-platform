"""
部署前检查脚本
在本地运行此脚本检查项目是否可以成功部署到 Streamlit
"""
import sys
import os

print("=" * 70)
print("🚀 EGFR 抑制剂预测平台 - 部署前检查")
print("=" * 70)

# 检查 Python 版本
print(f"\n📌 Python 版本: {sys.version}")
if sys.version_info < (3, 8):
    print("❌ 错误: 需要 Python 3.8 或更高版本")
    sys.exit(1)
else:
    print("✅ Python 版本符合要求")

# 检查关键文件
print("\n📁 检查关键文件...")
required_files = [
    "app.py",
    "requirements.txt",
    "packages.txt",
    "real_predictor.py",
    "fallback_predictor.py",
    "gnn_predictor.py",
    "feature_names.json",
    "rf_egfr_model_final.pkl",
    "gcn_egfr_best_model.pth",
]

all_files_exist = True
for file in required_files:
    exists = os.path.exists(file)
    status = "✅" if exists else "❌"
    print(f"   {status} {file}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n⚠️ 警告: 部分文件缺失，部署可能会失败")
else:
    print("\n✅ 所有关键文件都存在")

# 检查 requirements.txt
print("\n📦 检查依赖配置...")
try:
    with open("requirements.txt", "r") as f:
        req_content = f.read()
    
    # 检查关键依赖
    critical_deps = ["streamlit", "numpy", "scikit-learn", "setuptools", "ipywidgets"]
    for dep in critical_deps:
        if dep in req_content:
            print(f"   ✅ 包含 {dep}")
        else:
            print(f"   ⚠️ 缺少 {dep}")
    
    # 检查 numpy 版本限制
    if "numpy<2" in req_content:
        print("   ✅ numpy 版本限制正确 (<2)")
    else:
        print("   ⚠️ 建议添加 numpy<2 限制")
    
    # 检查 rich 版本
    if "rich<14" in req_content or "rich==13" in req_content:
        print("   ✅ rich 版本限制正确")
    else:
        print("   ⚠️ 建议限制 rich 版本 <14")
        
except Exception as e:
    print(f"   ❌ 无法读取 requirements.txt: {e}")

# 检查模型文件大小
print("\n💾 检查模型文件...")
model_file = "rf_egfr_model_final.pkl"
if os.path.exists(model_file):
    size_mb = os.path.getsize(model_file) / (1024 * 1024)
    print(f"   rf_egfr_model_final.pkl: {size_mb:.2f} MB")
    if size_mb > 100:
        print("   ⚠️ 警告: 模型文件较大，可能影响加载速度")

# 测试导入关键模块
print("\n🔧 测试关键模块导入...")

# 先测试 setuptools（解决 pkg_resources 问题）
try:
    import pkg_resources
    print("   ✅ pkg_resources 可用")
except ImportError:
    print("   ❌ pkg_resources 不可用，请安装 setuptools")

# 测试 numpy
try:
    import numpy as np
    print(f"   ✅ numpy {np.__version__}")
    if int(np.__version__.split('.')[0]) >= 2:
        print("   ⚠️ 警告: numpy 2.x 可能与保存的模型不兼容")
except ImportError:
    print("   ❌ numpy 未安装")

# 测试 sklearn
try:
    import sklearn
    print(f"   ✅ scikit-learn {sklearn.__version__}")
except ImportError:
    print("   ❌ scikit-learn 未安装")

# 测试 rdkit
try:
    from rdkit import Chem
    print("   ✅ RDKit 可用")
except ImportError:
    print("   ❌ RDKit 未安装")

# 测试模型加载
print("\n🎯 测试模型加载...")
try:
    import joblib
    model = joblib.load("rf_egfr_model_final.pkl")
    print(f"   ✅ 模型加载成功: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ 模型加载失败: {e}")
    print("   💡 提示: 运行 'python rebuild_model_for_deploy.py' 重建模型")

# 检查备用预测器
print("\n🔧 检查备用预测器...")
try:
    from fallback_predictor import FallbackEGFRPredictor
    predictor = FallbackEGFRPredictor()
    print("   ✅ 备用预测器可用")
except Exception as e:
    print(f"   ❌ 备用预测器错误: {e}")

print("\n" + "=" * 70)
print("📋 检查结果总结")
print("=" * 70)
print("""
如果检查发现问题，请按以下步骤修复:

1. 依赖问题:
   pip install -r requirements.txt

2. 模型不兼容问题 (No module named 'numpy._core'):
   python rebuild_model_for_deploy.py

3. 如果在 Streamlit Cloud 部署:
   - 确保 requirements.txt 包含 setuptools
   - 确保 rich 版本限制为 <14
   - 确保 ipywidgets>=8

4. 重新部署:
   git add .
   git commit -m "修复部署问题"
   git push origin main
""")
