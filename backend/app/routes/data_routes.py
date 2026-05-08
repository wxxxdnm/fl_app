from flask import Blueprint, jsonify, request
from ..services.data_manager import DataManager
from ..services.activity_service import activity_service
from .utils import get_json_body
import logging

logger = logging.getLogger(__name__)
data_bp = Blueprint('data', __name__)
data_manager = DataManager()

@data_bp.route('/datasets', methods=['GET'])
def get_available_datasets():
    """获取可用的数据集列表"""
    try:
        datasets = ['mnist', 'cifar10', 'cifar100']
        return jsonify({'datasets': datasets}), 200
    except Exception as e:
        logger.error(f"Error getting datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/datasets/<dataset_name>/info', methods=['GET'])
def get_dataset_info(dataset_name):
    """获取数据集信息"""
    try:
        info = data_manager.get_dataset_info(dataset_name)
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        return jsonify({'error': str(e)}), 404

@data_bp.route('/datasets/<dataset_name>/load', methods=['POST'])
def load_dataset(dataset_name):
    """加载数据集"""
    try:
        data, error_response = get_json_body()
        if error_response:
            return error_response
        try:
            batch_size = int(data.get('batch_size', 64))
        except (TypeError, ValueError):
            return jsonify({'error': 'batch_size must be an integer'}), 400
        train = data.get('train', True)
        if batch_size < 1:
            return jsonify({'error': 'batch_size must be at least 1'}), 400

        if dataset_name == 'mnist':
            loader = data_manager.load_mnist(train=train, batch_size=batch_size)
        elif dataset_name == 'cifar10':
            loader = data_manager.load_cifar10(train=train, batch_size=batch_size)
        elif dataset_name == 'cifar100':
            loader = data_manager.load_cifar100(train=train, batch_size=batch_size)
        else:
            return jsonify({'error': 'Unsupported dataset'}), 400

        activity_service.add_activity(f"数据集 {dataset_name} 加载完成", "info")
        return jsonify({
            'message': f'{dataset_name} loaded successfully',
            'batch_size': batch_size,
            'num_batches': len(loader)
        }), 200
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/federated/setup', methods=['POST'])
def setup_federated_data():
    """设置联邦学习数据集"""
    try:
        data, error_response = get_json_body()
        if error_response:
            return error_response
        dataset_name = data.get('dataset_name')
        try:
            num_clients = int(data.get('num_clients', 10))
            batch_size = int(data.get('batch_size', 64))
        except (TypeError, ValueError):
            return jsonify({'error': 'num_clients and batch_size must be integers'}), 400
        iid = data.get('iid', True)
        if not dataset_name:
            return jsonify({'error': 'dataset_name is required'}), 400
        if num_clients < 1:
            return jsonify({'error': 'num_clients must be at least 1'}), 400
        if batch_size < 1:
            return jsonify({'error': 'batch_size must be at least 1'}), 400

        dataloaders = data_manager.create_federated_datasets(
            dataset_name=dataset_name,
            num_clients=num_clients,
            batch_size=batch_size,
            iid=iid
        )

        client_info = []
        for i in range(num_clients):
            train_key = f'client_{i}_train'
            test_key = f'client_{i}_test'
            client_info.append({
                'client_id': i,
                'train_batches': len(dataloaders[train_key]),
                'test_batches': len(dataloaders[test_key])
            })

        activity_service.add_activity(f"联邦数据集分配完成：{dataset_name}, {num_clients} 个客户端", "success")
        return jsonify({
            'message': 'Federated data setup completed',
            'num_clients': num_clients,
            'iid': iid,
            'clients': client_info
        }), 200
    except Exception as e:
        logger.error(f"Error setting up federated data: {str(e)}")
        return jsonify({'error': str(e)}), 500
