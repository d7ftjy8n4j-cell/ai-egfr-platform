"""
检查哪个包依赖了rich 14.3.2
"""

import subprocess

print("=" * 70)
print("Checking which package depends on rich 14.3.2")
print("=" * 70)

# 检查markdown-it-py的依赖
print("\n1. Checking markdown-it-py dependencies...")
try:
    result = subprocess.run(
        ["pip", "show", "markdown-it-py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")

# 检查pygments的依赖
print("\n2. Checking pygments dependencies...")
try:
    result = subprocess.run(
        ["pip", "show", "pygments"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 70)
print("Solution")
print("=" * 70)
print("\nThe conflict is likely caused by markdown-it-py installing rich 14.3.2")
print("\nTo fix this, we need to:")
print("  1. Pin markdown-it-py to an older version")
print("  2. Or install rich first, then let pip resolve")
print("\nLet's try installing requirements with pip's --no-deps flag first")
