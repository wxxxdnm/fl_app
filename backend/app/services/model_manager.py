import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from .custom_dataset_manager import custom_dataset_manager
import logging
import os

logger = logging.getLogger(__name__)

class MNISTNet(nn.Module):
    """适用于MNIST的CNN模型"""
    def __init__(self, num_classes=10, **kwargs):
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
    def __init__(self, num_classes=10, **kwargs):
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
    def __init__(self, num_classes=100, **kwargs):
        super(CIFAR100Net, self).__init__(num_classes=num_classes, **kwargs)

class MLPNet(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 28, 28), **kwargs):
        super(MLPNet, self).__init__()
        input_features = 1
        for dim in input_shape:
            input_features *= dim
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(self.flatten(x))

class LeNetNet(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 28, 28), **kwargs):
        super(LeNetNet, self).__init__()
        input_channels = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class CIFARDeepCNN(nn.Module):
    def __init__(self, num_classes=10, input_shape=(3, 32, 32), **kwargs):
        super(CIFARDeepCNN, self).__init__()
        input_channels = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)

class CIFARResNet(nn.Module):
    def __init__(self, num_classes=10, input_shape=(3, 32, 32), **kwargs):
        super(CIFARResNet, self).__init__()
        input_channels = input_shape[0]
        self.in_channels = 32
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ModelManager:
    """模型管理器，负责模型的创建、配置和管理"""
    def __init__(self):
        self.available_models = {
            'mnist': ['cnn', 'mlp', 'lenet'],
            'cifar10': ['cnn', 'deep_cnn', 'resnet', 'mlp'],
            'cifar100': ['cnn', 'deep_cnn', 'resnet', 'mlp']
        }
        self.model_configs = {
            'mnist': {
                'input_shape': (1, 28, 28),
                'num_classes': 10,
                'default_model': 'cnn',
                'models': {
                    'cnn': {'label': 'CNN', 'model_class': MNISTNet},
                    'mlp': {'label': 'MLP', 'model_class': MLPNet},
                    'lenet': {'label': 'LeNet', 'model_class': LeNetNet}
                }
            },
            'cifar10': {
                'input_shape': (3, 32, 32),
                'num_classes': 10,
                'default_model': 'cnn',
                'models': {
                    'cnn': {'label': 'CNN', 'model_class': CIFAR10Net},
                    'deep_cnn': {'label': 'Deep CNN', 'model_class': CIFARDeepCNN},
                    'resnet': {'label': 'Small ResNet', 'model_class': CIFARResNet},
                    'mlp': {'label': 'MLP', 'model_class': MLPNet}
                }
            },
            'cifar100': {
                'input_shape': (3, 32, 32),
                'num_classes': 100,
                'default_model': 'cnn',
                'models': {
                    'cnn': {'label': 'CNN', 'model_class': CIFAR100Net},
                    'deep_cnn': {'label': 'Deep CNN', 'model_class': CIFARDeepCNN},
                    'resnet': {'label': 'Small ResNet', 'model_class': CIFARResNet},
                    'mlp': {'label': 'MLP', 'model_class': MLPNet}
                }
            }
        }

    def create_model(self, dataset_name: str, model_name: str = None) -> nn.Module:
        """创建模型实例"""
        if dataset_name not in self.model_configs and not custom_dataset_manager.is_custom_dataset(dataset_name):
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        config = self._get_model_config(dataset_name)
        selected_model = model_name or config['default_model']
        if selected_model not in config['models']:
            supported = ', '.join(config['models'].keys())
            raise ValueError(f"Unsupported model for {dataset_name}: {selected_model}. Supported: {supported}")

        model_spec = config['models'][selected_model]
        model = model_spec['model_class'](
            num_classes=config['num_classes'],
            input_shape=config['input_shape']
        )
        model.dataset_name = dataset_name
        model.model_name = selected_model
        model.input_shape = config['input_shape']
        model.num_classes = config['num_classes']

        logger.info(f"Created model for {dataset_name}/{selected_model}: {model.__class__.__name__}")
        return model

    def get_model_config(self, dataset_name: str) -> Dict:
        """获取模型配置信息"""
        if dataset_name not in self.model_configs and not custom_dataset_manager.is_custom_dataset(dataset_name):
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        config = self._get_model_config(dataset_name)
        return {
            'input_shape': config['input_shape'],
            'num_classes': config['num_classes'],
            'default_model': config['default_model'],
            'models': self.get_available_models(dataset_name)
        }

    def get_available_models(self, dataset_name: str = None):
        """获取可用的模型列表"""
        if dataset_name is None:
            return {
                name: self.get_available_models(name)
                for name in list(self.model_configs.keys()) + custom_dataset_manager.get_dataset_names()
            }
        if dataset_name not in self.model_configs and not custom_dataset_manager.is_custom_dataset(dataset_name):
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        config = self._get_model_config(dataset_name)
        return [
            {
                'value': model_name,
                'label': model_spec['label'],
                'model_class': model_spec['model_class'].__name__
            }
            for model_name, model_spec in config['models'].items()
        ]

    def _get_model_config(self, dataset_name: str) -> Dict:
        if dataset_name in self.model_configs:
            return self.model_configs[dataset_name]
        custom_info = custom_dataset_manager.get_dataset_info(dataset_name)
        return {
            'input_shape': tuple(custom_info['input_shape']),
            'num_classes': custom_info['num_classes'],
            'default_model': 'mlp',
            'models': {
                'mlp': {'label': 'MLP', 'model_class': MLPNet}
            }
        }

    def save_model(self, model: nn.Module, path: str, metadata: Dict = None):
        """保存模型及其元数据"""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        original_device = next(model.parameters()).device
        state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        save_dict = {
            'model_state_dict': state_dict,
            'model_class': model.__class__.__name__,
            'dataset_name': getattr(model, 'dataset_name', None),
            'model_name': getattr(model, 'model_name', None),
            'input_shape': getattr(model, 'input_shape', None),
            'num_classes': getattr(model, 'num_classes', None)
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
            'CIFAR100Net': CIFAR100Net,
            'MLPNet': MLPNet,
            'LeNetNet': LeNetNet,
            'CIFARDeepCNN': CIFARDeepCNN,
            'CIFARResNet': CIFARResNet
        }

        if model_class_name not in model_mapping:
            raise ValueError(f"Unknown model class: {model_class_name}")

        dataset_name = checkpoint.get('dataset_name')
        model_name = checkpoint.get('model_name')
        if dataset_name and model_name:
            model = self.create_model(dataset_name, model_name)
        else:
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
