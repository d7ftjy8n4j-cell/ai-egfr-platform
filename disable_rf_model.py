"""
临时禁用随机森林模型加载，便于首次部署
运行后app.py将只使用GNN模型
"""

import os

print("=" * 60)
print("临时禁用随机森林模型加载")
print("=" * 60)

app_py_path = 'app.py'

# 读取app.py
with open(app_py_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 查找并修改RF_PREDICTOR_AVAILABLE的定义
if "RF_PREDICTOR_AVAILABLE = True" in content:
    content = content.replace(
        "RF_PREDICTOR_AVAILABLE = True",
        "RF_PREDICTOR_AVAILABLE = False  # 临时禁用RF模型，待部署成功后重新启用"
    )
    print("✅ 已将 RF_PREDICTOR_AVAILABLE 设为 False")
elif "RF_PREDICTOR_AVAILABLE = False" in content:
    # 如果已经是False，添加注释
    content = content.replace(
        "RF_PREDICTOR_AVAILABLE = False",
        "RF_PREDICTOR_AVAILABLE = False  # 临时禁用RF模型，待部署成功后重新启用"
    )
    print("✅ RF_PREDICTOR_AVAILABLE 已经是 False")
else:
    print("⚠️  未找到 RF_PREDICTOR_AVAILABLE 的定义")

# 写回文件
with open(app_py_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n" + "=" * 60)
print("✅ 修改完成")
print("=" * 60)
print("\napp.py已更新，现在将只使用GNN模型进行预测")
print("首次部署成功后，可以重新启用RF模型")
