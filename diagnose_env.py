#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境诊断脚本 - 快速检查关键依赖包状态
"""

import sys
import subprocess

def check_package(package_name):
    """检查包是否已安装及其版本"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
        return None
    except:
        return None

def main():
    print("=" * 60)
    print("🔍 环境诊断报告")
    print("=" * 60)

    # 关键包列表
    key_packages = [
        ("streamlit", "1.29.0"),
        ("rich", "13.7.1"),
        ("markdown-it-py", "2.2.0"),
        ("pygments", "2.17.2"),
        ("ipywidgets", "7.6.3"),
        ("py3Dmol", "2.0.0.post2"),
        ("rdkit-pypi", "2022.9.5"),
        ("torch", "2.1.2+cpu"),
    ]

    # 可选包
    optional_packages = [
        ("plip", ">=2.2.0"),
        ("stmol", "❌ 不应存在"),
    ]

    print("\n【核心依赖】")
    print("-" * 60)
    all_ok = True
    for pkg, expected in key_packages:
        version = check_package(pkg)
        if version:
            # 简化版本比较
            expected_prefix = expected.split('+')[0].rsplit('.', 1)[0]
            version_prefix = version.split('+')[0].rsplit('.', 1)[0]
            if expected_prefix in version_prefix or version_prefix in expected_prefix:
                print(f"✅ {pkg:15} : {version}")
            else:
                print(f"⚠️  {pkg:15} : {version} (期望: {expected})")
                all_ok = False
        else:
            print(f"❌ {pkg:15} : 未安装")
            all_ok = False

    print("\n【可选依赖】")
    print("-" * 60)
    for pkg, expected in optional_packages:
        version = check_package(pkg)
        if pkg == "stmol":
            if version:
                print(f"❌ {pkg:15} : {version} (应移除!)")
                all_ok = False
            else:
                print(f"✅ {pkg:15} : 未安装 (正确)")
        elif version:
            print(f"✅ {pkg:15} : {version}")
        else:
            print(f"⚠️  {pkg:15} : 未安装 (降级模式)")

    print("\n【依赖冲突检查】")
    print("-" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ 无依赖冲突")
    else:
        print("❌ 发现依赖冲突:")
        print(result.stdout)

    print("\n" + "=" * 60)
    if all_ok:
        print("🎉 核心依赖状态正常，可以运行应用！")
        print("运行命令: streamlit run app.py")
    else:
        print("⚠️  检测到依赖问题")
        print("请运行修复脚本:")
        print("  Windows: fix_cloud_dependencies.bat")
        print("  Linux/Mac: bash fix_cloud_dependencies.sh")
    print("=" * 60)

if __name__ == "__main__":
    main()
