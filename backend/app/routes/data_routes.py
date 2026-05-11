from flask import Blueprint, jsonify, request
from ..services.data_manager import DataManager
from ..services.activity_service import activity_service
from ..services.custom_dataset_manager import custom_dataset_manager
from .utils import get_json_body, parse_bool
from werkzeug.utils import secure_filename
import datetime
import logging
import os

logger = logging.getLogger(__name__)
data_bp = Blueprint('data', __name__)
data_manager = DataManager()
AVAILABLE_DATASETS = ['mnist', 'cifar10', 'cifar100']
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uploads'))
ALLOWED_UPLOAD_EXTENSIONS = {'csv', 'json', 'jsonl', 'npz', 'npy', 'zip', 'tar', 'gz', 'pt', 'pth', 'pkl'}

os.makedirs(UPLOAD_DIR, exist_ok=True)

def _is_allowed_upload(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_UPLOAD_EXTENSIONS

def _uploaded_file_record(filename):
    path = os.path.join(UPLOAD_DIR, filename)
    stat = os.stat(path)
    record = {
        'filename': filename,
        'size_bytes': stat.st_size,
        'modified_time': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'path': path
    }
    custom_dataset = custom_dataset_manager.get_by_filename(filename)
    if custom_dataset:
        record['custom_dataset'] = custom_dataset
    return record

def get_all_dataset_names():
    return AVAILABLE_DATASETS + custom_dataset_manager.get_dataset_names()

@data_bp.route('/datasets', methods=['GET'])
def get_available_datasets():
    """获取可用的数据集列表"""
    try:
        return jsonify({'datasets': get_all_dataset_names()}), 200
    except Exception as e:
        logger.error(f"Error getting datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/uploads', methods=['GET'])
def list_uploaded_datasets():
    try:
        files = [
            _uploaded_file_record(filename)
            for filename in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, filename)) and filename != 'custom_datasets.json'
        ]
        files.sort(key=lambda item: item['modified_time'], reverse=True)
        return jsonify({'files': files}), 200
    except Exception as e:
        logger.error(f"Error listing uploaded datasets: {str(e)}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/uploads', methods=['POST'])
def upload_dataset_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'file is required'}), 400

        uploaded_file = request.files['file']
        if not uploaded_file.filename:
            return jsonify({'error': 'filename is required'}), 400

        filename = secure_filename(uploaded_file.filename)
        if not filename:
            return jsonify({'error': 'filename is invalid'}), 400
        if not _is_allowed_upload(filename):
            supported = ', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
            return jsonify({'error': f'Unsupported file type. Supported: {supported}'}), 400

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        saved_filename = f'{timestamp}_{filename}'
        save_path = os.path.join(UPLOAD_DIR, saved_filename)
        uploaded_file.save(save_path)
        custom_dataset = custom_dataset_manager.register_file(save_path)

        activity_service.add_activity(f"数据集文件 {filename} 上传完成", "success")
        file_record = _uploaded_file_record(saved_filename)
        if custom_dataset:
            file_record['custom_dataset'] = custom_dataset
        return jsonify({
            'message': 'Dataset file uploaded successfully',
            'file': file_record
        }), 201
    except Exception as e:
        logger.error(f"Error uploading dataset file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@data_bp.route('/uploads/<filename>', methods=['DELETE'])
def delete_uploaded_dataset(filename):
    try:
        safe_filename = secure_filename(filename)
        if not safe_filename:
            return jsonify({'error': 'filename is invalid'}), 400
        path = os.path.abspath(os.path.join(UPLOAD_DIR, safe_filename))
        if os.path.commonpath([UPLOAD_DIR, path]) != UPLOAD_DIR or not os.path.isfile(path):
            return jsonify({'error': 'Uploaded dataset file not found'}), 404
        custom_dataset_manager.unregister_file(safe_filename)
        os.remove(path)
        activity_service.add_activity(f"已删除上传数据集文件 {safe_filename}", "info")
        return jsonify({'message': 'Uploaded dataset file deleted'}), 200
    except Exception as e:
        logger.error(f"Error deleting uploaded dataset file: {str(e)}")
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
        elif custom_dataset_manager.is_custom_dataset(dataset_name):
            dataset = custom_dataset_manager.load_dataset(dataset_name)
            loader = data_manager.dataloaders[f"{dataset_name}_{'train' if train else 'test'}"] = __import__('torch').utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=train
            )
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
            non_iid_alpha = float(data.get('non_iid_alpha', 0.5))
            non_iid_seed = int(data.get('non_iid_seed', 42))
        except (TypeError, ValueError):
            return jsonify({'error': 'Federated data numeric parameters are invalid'}), 400
        try:
            iid = parse_bool(data.get('iid'), True)
        except ValueError as error:
            return jsonify({'error': f'iid {str(error)}'}), 400
        if not dataset_name:
            return jsonify({'error': 'dataset_name is required'}), 400
        if num_clients < 1:
            return jsonify({'error': 'num_clients must be at least 1'}), 400
        if batch_size < 1:
            return jsonify({'error': 'batch_size must be at least 1'}), 400
        if non_iid_alpha <= 0:
            return jsonify({'error': 'non_iid_alpha must be greater than 0'}), 400

        dataloaders = data_manager.create_federated_datasets(
            dataset_name=dataset_name,
            num_clients=num_clients,
            batch_size=batch_size,
            iid=iid,
            non_iid_alpha=non_iid_alpha,
            non_iid_seed=non_iid_seed
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
            'non_iid_alpha': non_iid_alpha,
            'non_iid_seed': non_iid_seed,
            'clients': client_info
        }), 200
    except Exception as e:
        logger.error(f"Error setting up federated data: {str(e)}")
        return jsonify({'error': str(e)}), 500
