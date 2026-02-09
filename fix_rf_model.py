"""
修复RF模型以兼容云端numpy 1.24.4
在本地运行此脚本来创建兼容的模型文件
"""

import os
import sys
import subprocess

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"[DIR] Working directory: {os.getcwd()}")

print("=" * 70)
print("RF Model Compatibility Fix Tool")
print("=" * 70)

# Check current numpy version
import numpy as np
print(f"\n[INFO] Current numpy version: {np.__version__}")

# Cloud target version
TARGET_NUMPY = "1.24.4"
print(f"[TARGET] Cloud numpy version: {TARGET_NUMPY}")

if np.__version__ == TARGET_NUMPY:
    print("[OK] Numpy version matches cloud, can use existing model")
else:
    print(f"[WARN] Numpy version mismatch")
    print(f"[INFO] Recommended: pip install numpy=={TARGET_NUMPY}")

# Check model file
print(f"\n[INFO] Checking model file...")
model_path = "rf_egfr_model_final.pkl"
if not os.path.exists(model_path):
    print(f"[ERROR] Model file not found: {model_path}")
    sys.exit(1)

size_mb = os.path.getsize(model_path) / (1024*1024)
print(f"[OK] Original model: {model_path} ({size_mb:.2f} MB)")

# Option 1: Install compatible numpy
print(f"\n" + "=" * 70)
print("Step 1: Installing compatible numpy and rebuilding model")
print("=" * 70)

print(f"\n[INFO] Installing numpy {TARGET_NUMPY}...")
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", f"numpy=={TARGET_NUMPY}"],
        check=True,
        capture_output=True
    )
    print(f"[OK] numpy {TARGET_NUMPY} installed successfully")

    # Reload numpy to use new version
    import importlib
    importlib.reload(np)
    print(f"[OK] Numpy version: {np.__version__}")

except subprocess.CalledProcessError as e:
    print(f"[ERROR] Installation failed: {e}")
    print(f"\n[INFO] You can manually run: pip install numpy=={TARGET_NUMPY}")
    sys.exit(1)

# Load original model
print(f"\n[INFO] Loading original model...")
try:
    import joblib
    model = joblib.load(model_path)
    print(f"[OK] Original model loaded successfully")
    print(f"[INFO] Model type: {type(model).__name__}")
except Exception as e:
    print(f"[ERROR] Failed to load original model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save compatible version
print(f"\n[INFO] Saving compatible model...")
compatible_model_path = "rf_egfr_model_compatible.pkl"
try:
    joblib.dump(model, compatible_model_path)
    print(f"[OK] Compatible model saved: {compatible_model_path}")

    # Verify can reload
    test_model = joblib.load(compatible_model_path)
    print(f"[OK] Compatible model verification successful")

    # Show file info
    size_mb = os.path.getsize(compatible_model_path) / (1024*1024)
    print(f"[INFO] File size: {size_mb:.2f} MB")

except Exception as e:
    print(f"[ERROR] Save failed: {e}")
    sys.exit(1)

# Test prediction
print(f"\n[INFO] Testing model prediction...")
try:
    if hasattr(model, 'n_features_in_'):
        test_data = np.zeros((1, model.n_features_in_))
        pred = model.predict(test_data)
        print(f"[OK] Model can predict: {pred}")
except Exception as e:
    print(f"[WARN] Prediction test error: {e}")

# Summary
print(f"\n" + "=" * 70)
print("Fix Complete")
print("=" * 70)

print(f"\n[OK] Created compatible model: {compatible_model_path}")
print(f"[OK] This model is compatible with numpy {TARGET_NUMPY}")

print(f"\n[NEXT STEPS]")
print(f"   1. git add {compatible_model_path}")
print(f"   2. git commit -m 'Add RF model compatible with numpy {TARGET_NUMPY}'")
print(f"   3. git push")
print(f"   4. Streamlit Cloud will automatically redeploy")

print(f"\n" + "=" * 70)
