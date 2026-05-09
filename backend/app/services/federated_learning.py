import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
import time
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FLClient:
    def __init__(self, client_id: int, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: str):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.local_epochs = 5
        
        # 客户端元数据和性能指标
        self.status = 'active'
        self.compute_power = np.random.choice(['low', 'medium', 'high'])
        self.network_speed = np.random.choice(['poor', 'good', 'excellent'])
        self.data_quality = np.random.choice(['low', 'medium', 'high'])
        self.participation_count = 0
        self.total_training_time = 0.0
        self.last_activity = time.time()

    def train(self, global_state_dict: Dict = None, proximal_mu: float = 0.0) -> Dict:
        """在本地客户端训练模型"""
        start_time = time.time()
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                if global_state_dict is not None and proximal_mu > 0:
                    proximal_term = torch.zeros((), device=self.device)
                    for name, param in self.model.named_parameters():
                        proximal_term += torch.sum((param - global_state_dict[name].to(self.device)) ** 2)
                    loss = loss + (proximal_mu / 2.0) * proximal_term
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        metrics = {
            'loss': total_loss / max(1, len(self.train_loader)),
            'accuracy': correct / max(1, total),
            'num_samples': total
        }

        training_time = time.time() - start_time
        self.total_training_time += training_time
        self.participation_count += 1
        self.last_activity = time.time()

        logger.info(f"Client {self.client_id} - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")
        return metrics

    def evaluate(self) -> Dict:
        """评估客户端模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        metrics = {
            'loss': total_loss / max(1, len(self.test_loader)),
            'accuracy': correct / max(1, total),
            'num_samples': total
        }

        return metrics

class FederatedLearning:
    SUPPORTED_ALGORITHMS = {
        'fedavg': 'FedAvg',
        'fedprox': 'FedProx',
        'fedavgm': 'FedAvgM',
        'fedadam': 'FedAdam',
        'fedyogi': 'FedYogi',
        'fedadagrad': 'FedAdagrad',
    }

    def __init__(
        self,
        global_model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        aggregation_algorithm: str = 'fedavg',
        server_lr: float = 1.0,
        server_momentum: float = 0.9,
        proximal_mu: float = 0.01,
        adaptive_beta1: float = 0.9,
        adaptive_beta2: float = 0.99,
        adaptive_tau: float = 1e-3
    ):
        self.global_model = global_model.to(device)
        self.device = device
        self.clients: List[FLClient] = []
        self.global_metrics = {}
        self.history = []
        self.aggregation_algorithm = aggregation_algorithm.lower()
        if self.aggregation_algorithm not in self.SUPPORTED_ALGORITHMS:
            supported = ', '.join(sorted(self.SUPPORTED_ALGORITHMS))
            raise ValueError(f"Unsupported aggregation algorithm: {aggregation_algorithm}. Supported: {supported}")
        self.server_lr = server_lr
        self.server_momentum = server_momentum
        self.proximal_mu = proximal_mu
        self.adaptive_beta1 = adaptive_beta1
        self.adaptive_beta2 = adaptive_beta2
        self.adaptive_tau = adaptive_tau
        self.server_momentum_state = {}
        self.server_adaptive_state = {}

    def add_client(self, client: FLClient):
        """添加客户端到联邦学习系统"""
        self.clients.append(client)

    @classmethod
    def get_available_algorithms(cls) -> List[Dict]:
        return [
            {'value': key, 'label': label}
            for key, label in cls.SUPPORTED_ALGORITHMS.items()
        ]

    def fed_avg(self, client_updates: List[Dict]) -> Dict:
        """FedAvg 聚合算法"""
        total_samples = sum(client['num_samples'] for client in client_updates)
        if total_samples <= 0:
            raise ValueError("Cannot aggregate client updates with zero samples")
        weighted_updates = {}

        for client in client_updates:
            weight = client['num_samples'] / total_samples
            for key, value in client['state_dict'].items():
                if not torch.is_floating_point(value):
                    weighted_updates[key] = value.clone()
                    continue
                if key not in weighted_updates:
                    weighted_updates[key] = value * weight
                else:
                    weighted_updates[key] += value * weight

        return weighted_updates

    def aggregate(self, client_updates: List[Dict]) -> Dict:
        """Aggregate client models with the configured server-side algorithm."""
        averaged_state = self.fed_avg(client_updates)
        if self.aggregation_algorithm in ('fedavg', 'fedprox'):
            return averaged_state

        global_state = self.global_model.state_dict()
        new_state = {}
        for key, averaged_value in averaged_state.items():
            global_value = global_state[key]
            if not torch.is_floating_point(global_value):
                new_state[key] = averaged_value
                continue

            averaged_value = averaged_value.to(self.device)
            global_value = global_value.to(self.device)
            delta = averaged_value - global_value

            if self.aggregation_algorithm == 'fedavgm':
                velocity = self.server_momentum_state.get(key, torch.zeros_like(delta))
                velocity = self.server_momentum * velocity + delta
                self.server_momentum_state[key] = velocity.detach().clone()
                new_state[key] = global_value + self.server_lr * velocity
            elif self.aggregation_algorithm in ('fedadam', 'fedyogi', 'fedadagrad'):
                first_moment, second_moment = self.server_adaptive_state.get(
                    key,
                    (torch.zeros_like(delta), torch.zeros_like(delta))
                )
                first_moment = self.adaptive_beta1 * first_moment + (1 - self.adaptive_beta1) * delta
                delta_sq = delta * delta
                if self.aggregation_algorithm == 'fedadagrad':
                    second_moment = second_moment + delta_sq
                elif self.aggregation_algorithm == 'fedyogi':
                    second_moment = second_moment - (1 - self.adaptive_beta2) * delta_sq * torch.sign(second_moment - delta_sq)
                else:
                    second_moment = self.adaptive_beta2 * second_moment + (1 - self.adaptive_beta2) * delta_sq
                self.server_adaptive_state[key] = (
                    first_moment.detach().clone(),
                    second_moment.detach().clone()
                )
                new_state[key] = global_value + self.server_lr * first_moment / (torch.sqrt(second_moment) + self.adaptive_tau)
            else:
                new_state[key] = averaged_value

        return new_state

    def train_round(self, selected_clients: List[FLClient]) -> Dict:
        """执行一轮联邦学习训练"""
        logger.info(f"Starting training round with {len(selected_clients)} clients")
        if not selected_clients:
            raise ValueError("At least one client is required for a training round")

        client_updates = []
        global_state_for_prox = None
        if self.aggregation_algorithm == 'fedprox' and self.proximal_mu > 0:
            global_state_for_prox = {
                name: param.detach().clone()
                for name, param in self.global_model.named_parameters()
            }
        for client in selected_clients:
            # 将全局模型参数复制到客户端
            client.model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))

            # 客户端本地训练
            train_metrics = client.train(global_state_for_prox, self.proximal_mu)

            # 收集客户端更新
            client_updates.append({
                'client_id': client.client_id,
                'state_dict': copy.deepcopy(client.model.state_dict()),
                'num_samples': train_metrics['num_samples'],
                'metrics': train_metrics
            })

        # 聚合客户端更新
        aggregated_state = self.aggregate(client_updates)

        # 更新全局模型
        self.global_model.load_state_dict(aggregated_state)

        # 计算本轮全局指标
        global_metrics = self.evaluate_global_model(selected_clients)

        round_history = {
            'round': len(self.history) + 1,
            'aggregation_algorithm': self.aggregation_algorithm,
            'client_metrics': [update['metrics'] for update in client_updates],
            'global_metrics': global_metrics
        }
        self.history.append(round_history)

        logger.info(f"Round completed - Global Acc: {global_metrics['accuracy']:.4f}")
        return round_history

    def evaluate_global_model(self, clients: List[FLClient]) -> Dict:
        """评估全局模型在所有客户端上的性能"""
        self.global_model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for client in clients:
                for data, target in client.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.global_model(data)
                    loss = nn.CrossEntropyLoss()(output, target)

                    total_loss += loss.item()
                    _, predicted = output.max(1)
                    total_correct += predicted.eq(target).sum().item()
                    total_samples += target.size(0)

        total_batches = sum(len(client.test_loader) for client in clients)
        if total_batches <= 0 or total_samples <= 0:
            raise ValueError("Cannot evaluate global model without test samples")

        return {
            'loss': total_loss / total_batches,
            'accuracy': total_correct / total_samples,
            'num_samples': total_samples
        }

    def get_training_history(self) -> List[Dict]:
        """获取训练历史记录"""
        return self.history

    def save_model(self, path: str):
        """保存全局模型"""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.global_model.state_dict().items()
        }
        torch.save({
            'model_state_dict': state_dict,
            'model_class': self.global_model.__class__.__name__,
            'dataset_name': getattr(self.global_model, 'dataset_name', None),
            'model_name': getattr(self.global_model, 'model_name', None),
            'input_shape': getattr(self.global_model, 'input_shape', None),
            'num_classes': getattr(self.global_model, 'num_classes', None),
            'metadata': {
                'device': self.device,
                'num_clients': len(self.clients),
                'rounds': len(self.history),
                'aggregation_algorithm': self.aggregation_algorithm
            }
        }, path)

    def load_model(self, path: str):
        """加载全局模型"""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else checkpoint
        if state_dict is None:
            raise ValueError("Checkpoint does not contain model_state_dict")
        self.global_model.load_state_dict(state_dict)
        self.global_model.to(self.device)
