import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import logging
import os

logger = logging.getLogger(__name__)

class MNISTNet(nn.Module):
    """适用于MNIST的CNN模型"""
    def __init__(self, num_classes=10):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR10Net(nn.Module):
    """适用于CIFAR10的CNN模型"""
    def __init__(self, num_classes=10):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CIFAR100Net(CIFAR10Net):
    """适用于CIFAR100的CNN模型"""
    def __init__(self, num_classes=100):
        super(CIFAR100Net, self).__init__(num_classes=num_classes)

class ModelManager:
    """模型管理器，负责模型的创建、配置和管理"""
    def __init__(self):
        self.available_models = {
            'mnist': MNISTNet,
            'cifar10': CIFAR10Net,
            'cifar100': CIFAR100Net
        }
        self.model_configs = {
            'mnist': {
                'input_shape': (1, 28, 28),
                'num_classes': 10,
                'model_class': MNISTNet
            },
            'cifar10': {
                'input_shape': (3, 32, 32),
                'num_classes': 10,
                'model_class': CIFAR10Net
            },
            'cifar100': {
                'input_shape': (3, 32, 32),
                'num_classes': 100,
                'model_class': CIFAR100Net
            }
        }

    def create_model(self, dataset_name: str, model_name: str = None) -> nn.Module:
        """创建模型实例"""
        if dataset_name not in self.model_configs:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        config = self.model_configs[dataset_name]
        model = config['model_class'](num_classes=config['num_classes'])

        logger.info(f"Created model for {dataset_name}: {model.__class__.__name__}")
        return model

    def get_model_config(self, dataset_name: str) -> Dict:
        """获取模型配置信息"""
        if dataset_name not in self.model_configs:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        return self.model_configs[dataset_name]

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return list(self.model_configs.keys())

    def save_model(self, model: nn.Module, path: str, metadata: Dict = None):
        """保存模型及其元数据"""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        original_device = next(model.parameters()).device
        state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        save_dict = {
            'model_state_dict': state_dict,
            'model_class': model.__class__.__name__
        }
        model.to(original_device)

        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> nn.Module:
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        if not isinstance(checkpoint, dict) or 'model_class' not in checkpoint or 'model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint must contain 'model_class' and 'model_state_dict'")
        model_class_name = checkpoint['model_class']

        # 根据模型类名创建模型实例
        model_mapping = {
            'MNISTNet': MNISTNet,
            'CIFAR10Net': CIFAR10Net,
            'CIFAR100Net': CIFAR100Net
        }

        if model_class_name not in model_mapping:
            raise ValueError(f"Unknown model class: {model_class_name}")

        model = model_mapping[model_class_name]()
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded from {path}")
        return model

    def get_model_summary(self, model: nn.Module) -> Dict:
        """获取模型摘要信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'model_name': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layers': len(list(model.children()))
        }

    def create_optimizer(self, model: nn.Module, optimizer_name: str = 'SGD', lr: float = 0.01, **kwargs):
        """创建优化器"""
        optimizers = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'RMSprop': torch.optim.RMSprop
        }

        if optimizer_name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optimizer_class = optimizers[optimizer_name]
        return optimizer_class(model.parameters(), lr=lr, **kwargs)

    def create_criterion(self, criterion_name: str = 'CrossEntropyLoss'):
        """创建损失函数"""
        criteria = {
            'CrossEntropyLoss': nn.CrossEntropyLoss,
            'MSELoss': nn.MSELoss,
            'BCELoss': nn.BCELoss
        }

        if criterion_name not in criteria:
            raise ValueError(f"Unsupported criterion: {criterion_name}")

        return criteria[criterion_name]()
