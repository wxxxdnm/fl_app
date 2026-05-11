import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from .custom_dataset_manager import custom_dataset_manager
import os
from typing import Tuple, Dict
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.cifar100_dir = os.path.join(data_dir, "cifar100_cache")
        os.makedirs(self.cifar100_dir, exist_ok=True)

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
        self.eval_transforms = {
            'mnist': self.transforms['mnist'],
            'cifar10': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            'cifar100': transforms.Compose([
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
                transform=self.eval_transforms['cifar10']
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
                root=self.cifar100_dir,
                train=True,
                download=True,
                transform=self.transforms['cifar100']
            )

        if train:
            dataset = self.datasets['cifar100']
        else:
            dataset = datasets.CIFAR100(
                root=self.cifar100_dir,
                train=False,
                download=True,
                transform=self.eval_transforms['cifar100']
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
                                batch_size: int = 64, iid: bool = True,
                                non_iid_alpha: float = 0.5,
                                non_iid_seed: int = 42) -> Dict[str, DataLoader]:
        """创建联邦学习数据集（IID或非IID）"""
        if dataset_name == 'mnist':
            full_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transforms['mnist']
            )
            eval_dataset = datasets.MNIST(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.eval_transforms['mnist']
            )
        elif dataset_name == 'cifar10':
            full_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transforms['cifar10']
            )
            eval_dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.eval_transforms['cifar10']
            )
        elif dataset_name == 'cifar100':
            full_dataset = datasets.CIFAR100(
                root=self.cifar100_dir,
                train=True,
                download=True,
                transform=self.transforms['cifar100']
            )
            eval_dataset = datasets.CIFAR100(
                root=self.cifar100_dir,
                train=True,
                download=True,
                transform=self.eval_transforms['cifar100']
            )
        elif custom_dataset_manager.is_custom_dataset(dataset_name):
            full_dataset = custom_dataset_manager.load_dataset(dataset_name)
            eval_dataset = full_dataset
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if num_clients < 1:
            raise ValueError("num_clients must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if non_iid_alpha <= 0:
            raise ValueError("non_iid_alpha must be greater than 0")

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
            subsets = self._create_non_iid_split(
                full_dataset,
                num_clients,
                non_iid_alpha,
                non_iid_seed
            )

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
            test_indices = self._resolve_subset_indices(test_subset)
            eval_test_subset = Subset(eval_dataset, test_indices) if eval_dataset is not full_dataset else test_subset

            client_dataloaders[f'client_{i}_train'] = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True
            )
            client_dataloaders[f'client_{i}_test'] = DataLoader(
                eval_test_subset, batch_size=batch_size, shuffle=False
            )

        logger.info(f"Created federated dataset with {num_clients} clients")
        return client_dataloaders

    def _resolve_subset_indices(self, dataset):
        if isinstance(dataset, Subset):
            parent_indices = self._resolve_subset_indices(dataset.dataset)
            return [parent_indices[index] for index in dataset.indices]
        return list(range(len(dataset)))

    def _extract_labels(self, dataset):
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if torch.is_tensor(targets):
                targets = targets.tolist()
            return [int(target) for target in targets]
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
            if torch.is_tensor(labels):
                labels = labels.tolist()
            return [int(label) for label in labels]
        return [int(label) for _, label in dataset]

    def _create_non_iid_split(self, dataset, num_clients: int,
                              alpha: float = 0.5,
                              seed: int = 42):
        """创建Non-IID数据划分"""
        # 按类别分组
        classes = {}
        labels = self._extract_labels(dataset)
        for idx, label in enumerate(labels):
            if label not in classes:
                classes[label] = []
            classes[label].append(idx)

        class_labels = sorted(classes.keys())
        rng = np.random.default_rng(seed)
        min_client_samples = 2
        if len(dataset) < num_clients * min_client_samples:
            raise ValueError("num_clients is too large for Non-IID split")

        client_data = [[] for _ in range(num_clients)]
        for _ in range(20):
            client_data = [[] for _ in range(num_clients)]
            for class_label in class_labels:
                class_indices = np.array(classes[class_label], dtype=np.int64)
                rng.shuffle(class_indices)
                proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=float))
                counts = rng.multinomial(len(class_indices), proportions)
                split_points = np.cumsum(counts)[:-1]
                for client_id, split_indices in enumerate(np.split(class_indices, split_points)):
                    client_data[client_id].extend(split_indices.tolist())
            if min(len(indices) for indices in client_data) >= min_client_samples:
                break

        for client_id in sorted(range(num_clients), key=lambda index: len(client_data[index])):
            while len(client_data[client_id]) < min_client_samples:
                donor_id = max(range(num_clients), key=lambda index: len(client_data[index]))
                if len(client_data[donor_id]) <= min_client_samples:
                    break
                donor_position = int(rng.integers(len(client_data[donor_id])))
                client_data[client_id].append(client_data[donor_id].pop(donor_position))

        subsets = []
        for indices in client_data:
            if not indices:
                raise ValueError("Non-IID split produced an empty client dataset")
            rng.shuffle(indices)
            subsets.append(Subset(dataset, indices))
        return subsets

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """获取数据集信息"""
        if dataset_name == 'mnist':
            dataset = datasets.MNIST(root=self.data_dir, train=True, download=True)
            return {
                'id': 'mnist',
                'name': 'MNIST',
                'num_samples': len(dataset),
                'num_classes': 10,
                'input_shape': (1, 28, 28),
                'classes': list(range(10))
            }
        elif dataset_name == 'cifar10':
            dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True)
            return {
                'id': 'cifar10',
                'name': 'CIFAR10',
                'num_samples': len(dataset),
                'num_classes': 10,
                'input_shape': (3, 32, 32),
                'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            }
        elif dataset_name == 'cifar100':
            dataset = datasets.CIFAR100(root=self.cifar100_dir, train=True, download=True)
            return {
                'id': 'cifar100',
                'name': 'CIFAR100',
                'num_samples': len(dataset),
                'num_classes': 100,
                'input_shape': (3, 32, 32),
                'classes': dataset.classes
            }
        elif custom_dataset_manager.is_custom_dataset(dataset_name):
            return custom_dataset_manager.get_dataset_info(dataset_name)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
