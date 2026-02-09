"""
检查RF模型相关文件是否被Git跟踪
"""
import subprocess
import os

print("=" * 70)
print("🔍 检查RF模型文件Git状态")
print("=" * 70)

# 需要检查的文件
files_to_check = [
    "rf_egfr_model_final.pkl",
    "rf_egfr_model_compatible.pkl",
    "feature_names.json",
    "real_predictor.py"
]

print("\n📋 文件状态检查：\n")

tracked_files = []
untracked_files = []
missing_files = []

for filename in files_to_check:
    if not os.path.exists(filename):
        missing_files.append(filename)
        print(f"❌ {filename:30} - 文件不存在")
        continue

    # 检查是否被Git跟踪
    try:
        result = subprocess.run(
            ["git", "ls-files", filename],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        if result.stdout.strip():
            tracked_files.append(filename)
            print(f"✅ {filename:30} - 已被Git跟踪")
        else:
            untracked_files.append(filename)
            print(f"⚠️  {filename:30} - 未被Git跟踪")

    except Exception as e:
        print(f"❓ {filename:30} - 检查失败: {e}")

# 检查.gitignore
print("\n📄 检查.gitignore文件:\n")
if os.path.exists(".gitignore"):
    with open(".gitignore", 'r', encoding='utf-8') as f:
        gitignore_content = f.read()
        if "*.pkl" in gitignore_content or ".pkl" in gitignore_content:
            print("⚠️  警告：.gitignore中可能包含.pkl文件规则")
            print("   这会导致模型文件不被Git跟踪！")
        else:
            print("✅ .gitignore中没有.pkl文件规则")
else:
    print("ℹ️  .gitignore文件不存在")

# 总结
print("\n" + "=" * 70)
print("📊 总结")
print("=" * 70)

if missing_files:
    print(f"\n❌ 缺失的文件 ({len(missing_files)}):")
    for f in missing_files:
        print(f"   - {f}")

if untracked_files:
    print(f"\n⚠️  未被Git跟踪的文件 ({len(untracked_files)}):")
    print("   这些文件不会被推送到GitHub！")
    for f in untracked_files:
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"   - {f} ({size_mb:.2f} MB)")

if tracked_files:
    print(f"\n✅ 已被Git跟踪的文件 ({len(tracked_files)}):")
    for f in tracked_files:
        print(f"   - {f}")

# 提供解决方案
if untracked_files:
    print("\n" + "=" * 70)
    print("💡 解决方案")
    print("=" * 70)
    print("\n要将这些文件添加到Git并推送到GitHub，请运行：")
    print("\ngit add", " ".join(untracked_files))
    print("git commit -m '添加RF模型相关文件'")
    print("git push")

print("\n" + "=" * 70)
