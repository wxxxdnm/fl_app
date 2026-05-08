import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import os
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.transforms = {
            'mnist': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'cifar10': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            'cifar100': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        }

        self.datasets = {}
        self.dataloaders = {}

    def load_mnist(self, train: bool = True, batch_size: int = 64) -> DataLoader:
        """加载MNIST数据集"""
        if 'mnist' not in self.datasets:
            self.datasets['mnist'] = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transforms['mnist']
            )

        if train:
            dataset = self.datasets['mnist']
        else:
            # 使用测试集
            dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transforms['mnist']
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=2
        )

        key = f"mnist_{'train' if train else 'test'}"
        self.dataloaders[key] = dataloader
        return dataloader

    def load_cifar10(self, train: bool = True, batch_size: int = 64) -> DataLoader:
        """加载CIFAR10数据集"""
        if 'cifar10' not in self.datasets:
            self.datasets['cifar10'] = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transforms['cifar10']
            )

        if train:
            dataset = self.datasets['cifar10']
        else:
            dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=2
        )

        key = f"cifar10_{'train' if train else 'test'}"
        self.dataloaders[key] = dataloader
        return dataloader

    def load_cifar100(self, train: bool = True, batch_size: int = 64) -> DataLoader:
        """加载CIFAR100数据集"""
        if 'cifar100' not in self.datasets:
            self.datasets['cifar100'] = datasets.CIFAR100(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transforms['cifar100']
            )

        if train:
            dataset = self.datasets['cifar100']
        else:
            dataset = datasets.CIFAR100(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=2
        )

        key = f"cifar100_{'train' if train else 'test'}"
        self.dataloaders[key] = dataloader
        return dataloader

    def create_federated_datasets(self, dataset_name: str, num_clients: int,
                                batch_size: int = 64, iid: bool = True) -> Dict[str, DataLoader]:
        """创建联邦学习数据集（IID或非IID）"""
        if dataset_name == 'mnist':
            full_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transforms['mnist']
            )
        elif dataset_name == 'cifar10':
            full_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transforms['cifar10']
            )
        elif dataset_name == 'cifar100':
            full_dataset = datasets.CIFAR100(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transforms['cifar100']
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if num_clients < 1:
            raise ValueError("num_clients must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # 划分数据给各个客户端
        if iid:
            # IID划分：随机均匀分配
            base_size = len(full_dataset) // num_clients
            remainder = len(full_dataset) % num_clients
            lengths = [base_size + (1 if i < remainder else 0) for i in range(num_clients)]
            if any(length == 0 for length in lengths):
                raise ValueError("num_clients cannot exceed the number of dataset samples")
            subsets = random_split(full_dataset, lengths)
        else:
            # Non-IID划分：按类别划分
            subsets = self._create_non_iid_split(full_dataset, num_clients)

        # 创建DataLoader
        client_dataloaders = {}
        for i, subset in enumerate(subsets):
            # 为每个客户端创建训练和测试数据
            if len(subset) < 2:
                raise ValueError(f"Client {i} has too few samples to split into train/test sets")
            train_size = max(1, int(0.8 * len(subset)))
            test_size = len(subset) - train_size
            if test_size == 0:
                train_size -= 1
                test_size = 1
            train_subset, test_subset = random_split(subset, [train_size, test_size])

            client_dataloaders[f'client_{i}_train'] = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True
            )
            client_dataloaders[f'client_{i}_test'] = DataLoader(
                test_subset, batch_size=batch_size, shuffle=False
            )

        logger.info(f"Created federated dataset with {num_clients} clients")
        return client_dataloaders

    def _create_non_iid_split(self, dataset, num_clients: int):
        """创建Non-IID数据划分"""
        # 按类别分组
        classes = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in classes:
                classes[label] = []
            classes[label].append(idx)

        # 为每个客户端分配特定的类别
        client_data = [[] for _ in range(num_clients)]
        class_labels = list(classes.keys())
        num_classes_per_client = max(1, min(2, len(class_labels)))

        for client_id in range(num_clients):
            # 为每个客户端分配2个类别
            class_start = client_id % len(class_labels)
            assigned_classes = []
            for i in range(num_classes_per_client):
                assigned_classes.append((class_start + i) % len(class_labels))

            # 分配对应类别的数据
            for class_idx in assigned_classes:
                class_label = class_labels[class_idx]
                client_data[client_id].extend(classes[class_label])

        # Shuffle and shard each client's assigned classes so clients do not all receive identical data.
        generator = torch.Generator().manual_seed(42)
        subsets = []
        for indices in client_data:
            if not indices:
                raise ValueError("Non-IID split produced an empty client dataset")
            shuffled_order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled_indices = [indices[i] for i in shuffled_order]
            max_samples = max(1, len(dataset) // num_clients)
            subsets.append(Subset(dataset, shuffled_indices[:max_samples]))
        return subsets

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name == 'mnist':
            dataset = datasets.MNIST(root=self.data_dir, train=True, download=True)
            return {
                'name': 'MNIST',
                'num_samples': len(dataset),
                'num_classes': 10,
                'input_shape': (1, 28, 28),
                'classes': list(range(10))
            }
        elif dataset_name == 'cifar10':
            dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True)
            return {
                'name': 'CIFAR10',
                'num_samples': len(dataset),
                'num_classes': 10,
                'input_shape': (3, 32, 32),
                'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            }
        elif dataset_name == 'cifar100':
            dataset = datasets.CIFAR100(root=self.data_dir, train=True, download=True)
            return {
                'name': 'CIFAR100',
                'num_samples': len(dataset),
                'num_classes': 100,
                'input_shape': (3, 32, 32),
                'classes': dataset.classes
            }
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
