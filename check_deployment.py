"""
部署前的环境检查脚本
检查关键依赖和配置是否正确
"""

import sys
import os

print("=" * 70)
print("📋 EGFR抑制剂智能预测系统 - 部署前环境检查")
print("=" * 70)

# 检查Python版本
print(f"\n🔍 检查Python版本...")
print(f"   Python版本: {sys.version}")
if sys.version_info >= (3, 9) and sys.version_info < (3, 12):
    print("   ✅ Python版本符合要求 (3.9-3.11)")
else:
    print("   ⚠️  建议使用Python 3.9-3.11")

# 检查关键依赖
print("\n🔍 检查关键依赖包...")
required_packages = {
    'streamlit': None,
    'numpy': None,
    'pandas': None,
    'torch': None,
    'rdkit': None,
    'joblib': None
}

for package in required_packages:
    try:
        module = __import__(package)
        if hasattr(module, '__version__'):
            version = module.__version__
            required_packages[package] = version
            print(f"   ✅ {package}: {version}")
        else:
            print(f"   ✅ {package}: 已安装（版本未知）")
    except ImportError:
        print(f"   ❌ {package}: 未安装")
        required_packages[package] = None

# 检查numpy版本兼容性
if required_packages['numpy']:
    numpy_version = required_packages['numpy']
    if numpy_version.startswith('1.24'):
        print(f"   ✅ numpy版本 {numpy_version} 与云端环境兼容")
    elif numpy_version > '1.26':
        print(f"   ⚠️  numpy版本 {numpy_version} 可能导致模型加载问题")
        print(f"   💡 建议: pip install numpy==1.24.4")

# 检查模型文件
print("\n🔍 检查模型文件...")
model_files = [
    'rf_egfr_model_final.pkl',
    'rf_egfr_model_compatible.pkl',
    'gcn_egfr_complete_model.pth'
]

for model_file in model_files:
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"   ✅ {model_file} ({size_mb:.2f} MB)")
    else:
        if model_file == 'rf_egfr_model_compatible.pkl':
            print(f"   ⚠️  {model_file}: 不存在（建议运行 rebuild_model.py）")
        else:
            print(f"   ❌ {model_file}: 不存在")

# 检查requirements.txt
print("\n🔍 检查requirements.txt...")
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"   ✅ requirements.txt 存在")

        # 检查关键配置
        checks = [
            ('numpy==1.24.4', 'numpy版本已锁定'),
            ('--find-links https://data.pyg.org', '使用预编译的torch-geometric'),
            ('rich>=10.14.0,<14', 'rich版本已锁定')
        ]

        for pattern, description in checks:
            if pattern in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ⚠️  {description} 未配置")

# 检查app.py
print("\n🔍 检查app.py...")
if os.path.exists('app.py'):
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
        if 'RF_PREDICTOR_AVAILABLE = False' in content:
            print(f"   ✅ RF模型已禁用（便于首次部署）")
        elif 'RF_PREDICTOR_AVAILABLE = True' in content:
            print(f"   ⚠️  RF模型已启用（如首次部署建议禁用）")
        else:
            print(f"   ⚠️  未找到RF模型配置")

# 检查GNN预测器
print("\n🔍 检查GNN预测器...")
if os.path.exists('gnn_predictor.py'):
    print(f"   ✅ gnn_predictor.py 存在")
else:
    print(f"   ❌ gnn_predictor.py 不存在")

# 总结
print("\n" + "=" * 70)
print("📊 检查总结")
print("=" * 70)

issues = []

if required_packages['numpy'] and required_packages['numpy'] > '1.26':
    issues.append("numpy版本过高，可能导致云端部署失败")

if not os.path.exists('rf_egfr_model_compatible.pkl'):
    issues.append("缺少兼容的RF模型文件")

if 'RF_PREDICTOR_AVAILABLE = True' in open('app.py', 'r', encoding='utf-8').read():
    issues.append("RF模型已启用，首次部署建议先禁用")

if issues:
    print("\n⚠️  发现以下问题：")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print("\n💡 建议操作：")
    print("   1. 运行: python rebuild_model.py")
    print("   2. 运行: python disable_rf_model.py")
    print("   3. 提交代码并部署")
else:
    print("\n✅ 所有关键检查通过！")
    print("💡 可以提交代码到GitHub并部署到Streamlit Cloud了")

print("\n" + "=" * 70)
