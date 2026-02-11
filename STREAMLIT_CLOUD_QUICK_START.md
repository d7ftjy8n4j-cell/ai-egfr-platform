# 🚀 Streamlit Cloud 快速修复指南

## 一分钟解决依赖冲突

### 🎯 你只需要做 3 件事：

---

## ✅ 第一步：确保文件存在

确认项目中有以下文件：

```
ai-egfr-platform/
├── .streamlit/
│   └── setup.sh          ← 我们刚创建的修复脚本
├── app.py
├── requirements.txt
├── diagnose_env.py       ← 环境诊断工具
└── ...
```

---

## ✅ 第二步：提交到 GitHub

```bash
cd C:\Users\dadamingli\Desktop\ai-egfr-platform

# 添加修复脚本
git add .streamlit/setup.sh

# 如果之前提交过requirements.txt，不需要重复添加
# 如果还没提交，也添加它
git add requirements.txt

# 提交
git commit -m "修复 Streamlit Cloud 依赖冲突"

# 推送
git push
```

---

## ✅ 第三步：配置 Streamlit Cloud

1. 登录 [share.streamlit.io](https://share.streamlit.io)

2. 找到你的应用，点击 **"Settings"**（设置）

3. 滚动到 **"Advanced settings"**（高级设置）

4. 在 **"Pre-build commands"**（预构建命令）输入框中，输入：

```bash
bash .streamlit/setup.sh
```

5. 点击 **"Save"**（保存）

6. 回到应用页面，点击 **"Rerun"**（重新运行）

---

## 🎉 完成！

Streamlit Cloud 会自动：
1. 执行 `setup.sh` 脚本
2. 清理旧的冲突包
3. 安装正确版本的依赖
4. 启动应用

---

## 🔍 如何验证修复成功？

1. 应用正常打开
2. 输入 SMILES 字符串进行预测
3. 点击 **"🔍 环境诊断"** 查看所有包的版本
4. 确认没有依赖冲突

---

## ⚠️ 如果仍然失败？

### 方法 A：查看完整日志

在 Streamlit Cloud 上：
1. 点击应用的 **"Logs"** 标签
2. 查看错误信息
3. 截图发送给我

### 方法 B：手动命令

如果 `setup.sh` 无法自动执行，你可以：

1. 在 Streamlit Cloud 的 **"Advanced settings"** 中，删除 **"Pre-build commands"**
2. 在 **"Python packages"** 中手动添加：
   ```
   rich==13.7.1
   markdown-it-py==2.2.0
   pygments==2.17.2
   ipywidgets==7.6.3
   streamlit==1.29.0
   ```
3. 然后在 `requirements.txt` 中删除这些行（避免重复）
4. 保存并重新部署

### 方法 C：降级模式

如果某些包（如 `plip`）持续安装失败：

1. 在 `requirements.txt` 中注释掉有问题的包：
   ```txt
   # plip>=2.2.0  # 注释掉，启用降级模式
   ```
2. 应用会自动禁用相关功能，核心预测不受影响

---

## 📊 预期状态

修复后，你的应用应该是：

| 功能 | 状态 |
|------|------|
| 🧬 GNN 预测 | ✅ 正常 |
| 📊 数据可视化 | ✅ 正常 |
| 🎨 3D 分子结构 | ⚠️ 可能受限 |
| 🔬 PLIP 分析 | ⚠️ 可能受限（降级模式） |

---

## 💡 常见问题

**Q: 为什么不能直接用 `pip install -r requirements.txt`？**
A: 因为 Streamlit Cloud 环境中已经预装了旧版本的包，需要先清理再安装。

**Q: setup.sh 会在每次部署时都执行吗？**
A: 是的，这能确保每次部署时依赖都是正确的。

**Q: 如果我想在本地测试修复脚本？**
A: 运行 `bash .streamlit/setup.sh`，然后 `streamlit run app.py`

---

## 🎯 总结

记住这 3 个命令就够了：

```bash
# 1. 提交代码
git add .streamlit/setup.sh && git commit -m "修复依赖" && git push

# 2. 在 Streamlit Cloud 设置中添加预构建命令
bash .streamlit/setup.sh

# 3. 点击 Rerun
```

就这么简单！🎉
