# FL_app 联邦学习可视化平台

FL_app 是一个基于 `Flask + React + PyTorch` 的联邦学习实验与可视化平台，支持数据集管理、模型配置、联邦训练、客户端监控、训练历史记录和结果可视化，适合课程设计、实验演示、算法对比和答辩展示。

## 项目特点

- 支持内置数据集：`MNIST`、`CIFAR10`、`CIFAR100`
- 支持自定义上传可训练分类数据集
- 支持 `IID` / `Non-IID` 联邦数据划分
- 支持可配置 `Non-IID` 类别偏斜参数：每客户端类别数、随机种子
- 支持多种联邦聚合算法：
  - `FedAvg`
  - `FedProx`
  - `FedAvgM`
  - `FedAdam`
  - `FedYogi`
  - `FedAdagrad`
- 支持多模型选择：`CNN`、`MLP`、`LeNet`、`Deep CNN`、`Small ResNet`
- 支持按数据集筛选兼容模型，后端会校验输入形状、类别数和模型选择
- 支持训练状态监控、停止训练、指标曲线和训练历史持久化
- 支持准确率、损失、精确率、召回率、F1、平衡准确率、吞吐量等指标展示
- 支持客户端状态、客户端性能、类别分布、混淆矩阵等当前和历史可视化分析
- 支持模型保存、模型加载和历史模型记录

## 技术栈

- 后端：`Flask`、`Flask-CORS`、`PyTorch`、`TorchVision`
- 前端：`React 18`、`Ant Design`、`React Router`、`Recharts`、`Chart.js`
- 数据处理：`NumPy`、`Pandas`
- 可视化：`Matplotlib`、`Seaborn`
- 系统监控：`psutil`

## 目录结构

```text
FL_app/
├─ backend/
│  ├─ app/
│  │  ├─ routes/                 # Flask API 路由
│  │  └─ services/               # 数据、模型、训练、历史记录等核心逻辑
│  ├─ checkpoints/               # 模型权重保存目录
│  ├─ data/                      # 数据缓存、上传数据集、历史记录
│  └─ requirements.txt           # 后端依赖
├─ frontend/
│  ├─ public/
│  ├─ src/
│  │  ├─ components/             # 通用组件
│  │  └─ pages/                  # 页面组件
│  └─ package.json               # 前端依赖与脚本
├─ scripts/                      # 辅助脚本
├─ start.py                      # 一键启动脚本
├─ requirements.txt              # 根目录 Python 依赖，不包含 torch/torchvision
└─ README.md
```

## 运行环境

推荐环境：

- Python `3.10`
- Node.js `16+`
- npm `8+`
- 可选：NVIDIA GPU + CUDA `11.8`

推荐使用 Conda 虚拟环境：

```bash
conda create -n fl_app python=3.10 -y
conda activate fl_app
```

## 安装步骤

### 1. 安装 PyTorch

如果使用 GPU，推荐先安装 CUDA 11.8 对应版本：

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

或使用 pip：

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

如果只使用 CPU，可根据 PyTorch 官方文档安装 CPU 版本。

### 2. 安装 Python 依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

说明：

- 根目录 `requirements.txt` 不包含 `torch` / `torchvision`
- `backend/requirements.txt` 包含后端完整运行依赖，包括 `torch` / `torchvision`
- `python start.py` 会检查并在缺少关键依赖时使用 `backend/requirements.txt` 安装
- 为避免覆盖已安装的 GPU 版 PyTorch，建议先手动安装 PyTorch，再安装其余依赖

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

### 方式一：一键启动

在项目根目录执行：

```bash
python start.py
```

脚本会检查依赖并启动：

- 后端 API：`http://localhost:5000`
- 前端页面：`http://localhost:3000`

### 方式二：分别启动前后端

启动后端：

```bash
cd backend
set FLASK_APP=app/__init__.py
flask run --debug --port=5000
```

启动前端：

```bash
cd frontend
npm start
```

浏览器访问：

```text
http://localhost:3000
```

后端 API 根路径：

```text
http://localhost:5000
```

## 主要功能页面

- **首页仪表盘**：查看客户端数量、数据集数量、训练轮次、最新准确率、训练历史和模型记录
- **数据管理**：查看内置数据集，上传自定义数据集，配置联邦数据划分
- **模型训练**：配置数据集、模型、客户端数量、训练轮次、聚合算法、IID/Non-IID 参数并启动训练
- **客户端管理**：查看客户端状态、资源信息、参与次数和性能指标
- **可视化分析**：查看训练曲线、客户端表现、类别分布、混淆矩阵和综合指标

## 数据集说明

### 内置数据集

| 数据集 | 类别数 | 默认输入 | 支持模型 |
| --- | ---: | --- | --- |
| `mnist` | 10 | `1 x 28 x 28` | `cnn`、`mlp`、`lenet` |
| `cifar10` | 10 | `3 x 32 x 32` | `cnn`、`deep_cnn`、`resnet`、`mlp` |
| `cifar100` | 100 | `3 x 32 x 32` | `cnn`、`deep_cnn`、`resnet`、`mlp` |

内置数据集首次加载时会自动下载到 `backend/data/`，其中 `CIFAR100` 使用 `backend/data/cifar100_cache/` 缓存。

### 自定义数据集

数据管理页面支持上传文件。后端会将文件保存到：

```text
backend/data/uploads/
```

支持上传扩展名：

- 可注册为可训练数据集：`csv`、`json`、`jsonl`、`npy`、`npz`、`pt`、`pth`
- 可存储但不一定可训练：`zip`、`tar`、`gz`、`pkl`

自定义可训练数据集要求：

- 任务类型为监督分类
- 特征必须能转换为数值型
- 标签列优先识别 `label`、`target`、`y`，否则默认使用最后一列
- `npy` 文件要求二维数组，最后一列为标签
- `npz` / `pt` / `pth` 支持 `x/y`、`features/labels`、`data/targets` 等键名
- 至少包含 2 个样本和 2 个类别

自定义数据集训练时默认使用 `MLP` 模型。

### 数据集与模型兼容性

平台会根据所选数据集返回兼容模型列表，并在训练启动前进行后端校验：

- `mnist`：`cnn`、`mlp`、`lenet`
- `cifar10`：`cnn`、`deep_cnn`、`resnet`、`mlp`
- `cifar100`：`cnn`、`deep_cnn`、`resnet`、`mlp`
- 自定义数据集：默认使用 `mlp`

如果传入不兼容的数据集或模型组合，训练接口会返回明确的 `400` 错误信息。

## 联邦训练说明

训练流程：

1. 选择数据集和模型
2. 设置客户端数量、训练轮次、批大小和客户端参与比例
3. 选择 `IID` 或 `Non-IID` 数据划分
4. 选择聚合算法
5. 启动训练并实时查看状态和指标
6. 训练完成后查看历史记录、保存模型或进行可视化分析

### 支持的聚合算法

| 算法 | 参数说明 |
| --- | --- |
| `fedavg` | 基础加权平均 |
| `fedprox` | 支持近端项 `proximal_mu` |
| `fedavgm` | 支持服务端动量 `server_momentum` |
| `fedadam` | 支持自适应参数 `adaptive_beta1`、`adaptive_beta2`、`adaptive_tau` |
| `fedyogi` | 支持自适应参数 |
| `fedadagrad` | 支持自适应参数 |

### 训练指标

平台会记录和展示：

- `accuracy`
- `loss`
- `precision`
- `recall`
- `f1_score`
- `balanced_accuracy`
- `per_class_precision`
- `per_class_recall`
- `per_class_f1`
- `training_time`
- `evaluation_time`
- `samples_per_second`

## 后端接口概览

| 模块 | 接口 | 说明 |
| --- | --- | --- |
| 首页 | `GET /api/main/dashboard_stats` | 获取仪表盘统计、训练历史、模型记录和最近活动 |
| 数据 | `GET /api/data/datasets` | 获取可用数据集 |
| 数据 | `GET /api/data/uploads` | 获取上传文件列表 |
| 数据 | `POST /api/data/uploads` | 上传数据集文件 |
| 数据 | `DELETE /api/data/uploads/<filename>` | 删除上传数据集文件 |
| 数据 | `GET /api/data/datasets/<dataset_name>/info` | 获取数据集信息 |
| 数据 | `POST /api/data/datasets/<dataset_name>/load` | 加载数据集 |
| 数据 | `POST /api/data/federated/setup` | 创建联邦数据划分 |
| 模型 | `GET /api/model/models` | 获取可用模型 |
| 模型 | `GET /api/model/models/<dataset_name>/config` | 获取模型配置 |
| 模型 | `POST /api/model/models/<dataset_name>/create` | 创建模型 |
| 模型 | `POST /api/model/models/save` | 保存模型 |
| 模型 | `POST /api/model/models/load` | 加载模型 |
| 模型 | `GET /api/model/history` | 获取历史模型记录 |
| 训练 | `GET /api/train/algorithms` | 获取聚合算法列表 |
| 训练 | `POST /api/train/start` | 启动训练 |
| 训练 | `GET /api/train/status` | 查询训练状态 |
| 训练 | `POST /api/train/stop` | 停止训练 |
| 训练 | `POST /api/train/save` | 保存当前训练模型 |
| 训练 | `GET /api/train/metrics` | 获取训练指标 |
| 训练 | `DELETE /api/train/history/<run_id>` | 删除训练历史记录 |
| 客户端 | `GET /api/clients/` | 获取客户端列表 |
| 客户端 | `GET /api/clients/stats` | 获取客户端统计 |
| 客户端 | `GET /api/clients/performance` | 获取客户端性能监控数据 |
| 可视化 | `GET /api/viz/training_curves` | 获取训练曲线数据 |
| 可视化 | `GET /api/viz/model_performance` | 获取客户端模型表现 |
| 可视化 | `POST /api/viz/confusion_matrix` | 生成混淆矩阵数据 |
| 可视化 | `GET /api/viz/client_distribution` | 获取客户端数据分布 |

## 数据与模型保存位置

- 内置数据集缓存：`backend/data/`
- 上传数据集：`backend/data/uploads/`
- 自定义数据集注册信息：`backend/data/uploads/custom_datasets.json`
- 训练历史记录：`backend/data/history/`
- 模型权重：`backend/checkpoints/`

## 常见问题

### 1. 前端无法访问后端

请确认后端已启动，并检查 `5000` 端口是否被占用。

### 2. 数据集下载失败

请检查网络连接。如果自动下载失败，可手动下载对应数据集并放入 `backend/data/` 相关目录。

### 3. Torch 或 CUDA 不可用

检查 PyTorch 版本和 CUDA 状态：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

如果 `torch.cuda.is_available()` 为 `False`，训练接口会自动回退到 CPU。

### 4. npm 安装依赖失败

可尝试：

```bash
npm install --legacy-peer-deps
```

或清理前端依赖后重新安装。

### 5. 自定义数据集不能训练

请确认文件满足以下条件：

- 文件格式在可训练格式列表中
- 特征列均为数值
- 标签列存在或最后一列可作为标签
- 至少包含 2 个类别
- 没有 `NaN` 或无穷大数值

## 适用场景

- 联邦学习课程实验
- 联邦学习算法对比演示
- 数据分布与客户端差异可视化
- 模型训练流程展示
- 本科/研究生课程设计与答辩项目
