# 使用Python 3.9的官方镜像
FROM python:3.9-slim

# 安装系统依赖（解决RDKit问题）
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有应用文件
COPY . .

# 暴露端口
EXPOSE 8501

# 启动命令
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]