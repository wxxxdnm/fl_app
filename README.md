# FL_app 联邦学习可视化平台

这是一个基于 Flask + React 的联邦学习实验与可视化项目，提供数据集管理、联邦训练、客户端监控和结果可视化等功能，适合用于课程设计、实验演示和联邦学习流程验证。

## 项目特点

- 支持 `MNIST` 和 `CIFAR10` 数据集
- 支持 `IID` / `Non-IID` 联邦数据划分
- 支持多种聚合算法：
  - `FedAvg`
  - `FedProx`
  - `FedAvgM`
  - `FedAdam`
  - `FedYogi`
  - `FedAdagrad`
- 支持训练过程监控与指标展示
- 支持客户端状态与资源信息查看
- 支持训练曲线、类别分布、混淆矩阵等可视化分析

## 技术栈

- 后端：`Flask`、`PyTorch`、`TorchVision`
- 前端：`React 18`、`Ant Design`、`Recharts`
- 数据分析与可视化：`NumPy`、`Pandas`、`Matplotlib`、`Seaborn`

## 目录结构

```text
FL_app/
├─ backend/                 # Flask 后端
│  ├─ app/
│  │  ├─ routes/            # API 路由
│  │  └─ services/          # 数据、模型、联邦学习核心逻辑
│  ├─ checkpoints/          # 模型权重保存目录
│  └─ requirements.txt      # 原后端依赖文件
├─ frontend/                # React 前端
│  ├─ public/
│  └─ src/
│     ├─ components/
│     └─ pages/
├─ data/                    # 数据集下载与缓存目录
├─ logs/                    # 日志目录
├─ start.py                 # 一键启动脚本
├─ requirements.txt         # 根目录 Python 依赖
└─ README.md
```

## 运行环境

推荐使用 Conda 虚拟环境 `fl_app`：

```bash
conda create -n fl_app python=3.10 -y
conda activate fl_app
```

本项目默认按 `GPU + CUDA 11.8` 环境配置 PyTorch。

## 安装步骤

### 1. 安装 GPU 版 PyTorch（CUDA 11.8）

建议优先使用 Conda 安装，和当前环境更匹配。

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

如果你更想使用 `pip`，也可以使用 PyTorch 官方提供的 `cu118` 源：

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

说明：

- 上面两种方式二选一即可，不要混装
- `conda` 方案更适合你当前指定的 `fl_app` 虚拟环境
- 以上 CUDA 11.8 安装命令参考 PyTorch 官方文档

### 2. 安装其余 Python 依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

说明：

- 根目录 `requirements.txt` 不再包含 `torch` / `torchvision`
- 这样可以避免 `pip install -r requirements.txt` 把已安装的 GPU 版 PyTorch 覆盖掉

### 3. 安装前端依赖

```bash
cd frontend
npm install
```

如果出现依赖冲突，可使用：

```bash
npm install --legacy-peer-deps
```

## 启动方式

### 方式一：根目录一键启动

```bash
python start.py
```

默认会同时启动：

- 后端：`http://localhost:5000`
- 前端：`http://localhost:3000`

### 方式二：分别启动前后端

先启动后端：

```bash
cd backend
set FLASK_APP=app/__init__.py
flask run --debug --port=5000
```

再启动前端：

```bash
cd frontend
npm start
```

## 主要功能页面

- 首页仪表盘：查看训练概况、最近活动和核心统计信息
- 数据管理：加载数据集、查看数据集信息、配置联邦数据划分
- 模型训练：配置客户端数量、轮次、聚合算法并启动训练
- 客户端管理：查看客户端状态、性能指标和资源占用
- 可视化分析：查看训练曲线、模型表现、类别分布和混淆矩阵

## 后端接口概览

- `/api/main/dashboard_stats`：首页统计信息
- `/api/data/*`：数据集加载与联邦数据划分
- `/api/model/*`：模型创建、加载与保存
- `/api/train/*`：训练启动、状态查询、停止与指标获取
- `/api/clients/*`：客户端状态与性能信息
- `/api/viz/*`：可视化分析接口

## 数据与模型说明

- 数据集首次加载时会自动下载到根目录的 `data/` 下
- 训练完成后的模型默认保存在 `backend/checkpoints/`
- 项目当前内置的数据集为 `MNIST` 和 `CIFAR10`

## 常见问题

### 1. 前端无法访问后端

请确认后端是否已启动，并检查 `5000` 端口是否被占用。

### 2. 数据集下载失败

请检查网络连接，或手动将数据集下载到 `data/` 目录。

### 3. Torch 安装失败

请优先使用上面的 Conda 命令安装 CUDA 11.8 版本；如果仍需使用 pip，再执行官方 `cu118` 安装命令。

安装完成后可用下面的命令检查 CUDA 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## 适用场景

- 联邦学习课程实验
- 联邦学习算法对比演示
- 可视化展示训练过程
- 小型教学或答辩项目
