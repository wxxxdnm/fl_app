from flask import Blueprint, jsonify, request
from ..services.federated_learning import FLClient
from . import train_routes
from .utils import get_json_body
import logging
import datetime
import numpy as np
import psutil
import subprocess
import torch

logger = logging.getLogger(__name__)
client_bp = Blueprint('clients', __name__)
VALID_CLIENT_STATUSES = {'active', 'inactive', 'busy'}

# 存储静态客户端信息（如果没有训练正在运行）
static_clients = {}

def get_gpu_resources():
    """Return GPU availability and memory usage from PyTorch when CUDA is available."""
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'name': 'No CUDA GPU',
            'usage': 0,
            'memory': 0,
            'memory_used_mb': 0,
            'memory_total_mb': 0
        }

    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total,name',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            timeout=1,
            check=True
        )
        devices = []
        total_memory = 0.0
        used_memory = 0.0
        total_usage = 0.0

        for device_index, line in enumerate(result.stdout.strip().splitlines()):
            usage, memory_used, memory_total, name = [part.strip() for part in line.split(',', 3)]
            usage = float(usage)
            memory_used = float(memory_used)
            memory_total = float(memory_total)
            total_usage += usage
            used_memory += memory_used
            total_memory += memory_total
            devices.append({
                'id': device_index,
                'name': name,
                'usage': round(usage, 1),
                'memory_used_mb': round(memory_used, 1),
                'memory_total_mb': round(memory_total, 1),
                'memory_percent': round((memory_used / memory_total) * 100, 1) if memory_total else 0
            })

        if devices:
            memory_percent = round((used_memory / total_memory) * 100, 1) if total_memory else 0
            return {
                'available': True,
                'count': len(devices),
                'name': devices[0]['name'],
                'usage': round(total_usage / len(devices), 1),
                'memory': memory_percent,
                'memory_used_mb': round(used_memory, 1),
                'memory_total_mb': round(total_memory, 1),
                'devices': devices
            }
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    device_count = torch.cuda.device_count()
    devices = []
    total_memory = 0
    used_memory = 0

    for device_index in range(device_count):
        props = torch.cuda.get_device_properties(device_index)
        total = props.total_memory
        allocated = torch.cuda.memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        used = max(allocated, reserved)
        total_memory += total
        used_memory += used
        devices.append({
            'id': device_index,
            'name': props.name,
            'memory_used_mb': round(used / (1024 ** 2), 1),
            'memory_total_mb': round(total / (1024 ** 2), 1),
            'memory_percent': round((used / total) * 100, 1) if total else 0
        })

    memory_percent = round((used_memory / total_memory) * 100, 1) if total_memory else 0
    return {
        'available': True,
        'count': device_count,
        'name': devices[0]['name'] if devices else 'CUDA GPU',
        'usage': memory_percent,
        'memory': memory_percent,
        'memory_used_mb': round(used_memory / (1024 ** 2), 1),
        'memory_total_mb': round(total_memory / (1024 ** 2), 1),
        'devices': devices
    }

def get_current_clients():
    """从正在运行的训练系统中获取客户端，或者返回静态客户端"""
    if train_routes.fl_system is not None and train_routes.fl_system.clients:
        # 将 FLClient 对象转换为字典格式
        dynamic_clients = {}
        for client in train_routes.fl_system.clients:
            dynamic_clients[client.client_id] = {
                'client_id': client.client_id,
                'status': client.status,
                'compute_power': client.compute_power,
                'network_speed': client.network_speed,
                'data_quality': client.data_quality,
                'participation_count': client.participation_count,
                'avg_training_time': round(client.total_training_time / max(1, client.participation_count), 2),
                'last_activity': datetime.datetime.fromtimestamp(client.last_activity).isoformat()
            }
        return dynamic_clients
    return static_clients

@client_bp.route('/', methods=['GET'])
def get_all_clients():
    """获取所有客户端信息"""
    try:
        current_clients = get_current_clients()
        return jsonify({
            'num_clients': len(current_clients),
            'clients': current_clients
        }), 200
    except Exception as e:
        logger.error(f"Error getting clients: {str(e)}")
        return jsonify({'error': str(e)}), 500

@client_bp.route('/<int:client_id>', methods=['GET'])
def get_client(client_id):
    """获取特定客户端信息"""
    try:
        current_clients = get_current_clients()
        if client_id not in current_clients:
            return jsonify({'error': f'Client {client_id} not found'}), 404

        return jsonify(current_clients[client_id]), 200
    except Exception as e:
        logger.error(f"Error getting client {client_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@client_bp.route('/<int:client_id>/status', methods=['POST'])
def update_client_status(client_id):
    """更新客户端状态"""
    try:
        data, error_response = get_json_body()
        if error_response:
            return error_response
        status = data.get('status')
        if not status:
            return jsonify({'error': 'status is required'}), 400
        if status not in VALID_CLIENT_STATUSES:
            supported = ', '.join(sorted(VALID_CLIENT_STATUSES))
            return jsonify({'error': f'Unsupported status. Supported: {supported}'}), 400

        # 如果训练正在运行，更新 dynamic client
        if train_routes.fl_system is not None:
            for client in train_routes.fl_system.clients:
                if client.client_id == client_id:
                    client.status = status
                    return jsonify({'message': f'Client {client_id} status updated to {status}'}), 200

        # 否则更新静态客户端
        if client_id not in static_clients:
            static_clients[client_id] = {
                'client_id': client_id,
                'compute_power': 'medium',
                'network_speed': 'good',
                'data_quality': 'medium',
                'participation_count': 0,
                'avg_training_time': 0,
                'last_activity': datetime.datetime.now().isoformat()
            }
        static_clients[client_id]['status'] = status
        static_clients[client_id]['last_updated'] = datetime.datetime.now().isoformat()

        return jsonify({
            'message': f'Client {client_id} status updated to {status}'
        }), 200
    except Exception as e:
        logger.error(f"Error updating client status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@client_bp.route('/<int:client_id>/metrics', methods=['GET'])
def get_client_metrics(client_id):
    """获取客户端性能指标"""
    return get_client(client_id)

@client_bp.route('/stats', methods=['GET'])
def get_client_stats():
    """获取客户端统计信息"""
    try:
        current_clients = get_current_clients()
        total_clients = len(current_clients)
        active_clients = sum(1 for c in current_clients.values() if c.get('status') == 'active')
        busy_clients = sum(1 for c in current_clients.values() if c.get('status') == 'busy')
        inactive_clients = total_clients - active_clients - busy_clients

        # 计算平均性能指标
        if total_clients > 0:
            avg_participation = sum(c.get('participation_count', 0) for c in current_clients.values()) / total_clients
            avg_training_time = sum(float(c.get('avg_training_time', 0)) for c in current_clients.values()) / total_clients
        else:
            avg_participation = 0
            avg_training_time = 0

        return jsonify({
            'total_clients': total_clients,
            'active_clients': active_clients,
            'busy_clients': busy_clients,
            'inactive_clients': inactive_clients,
            'avg_participation': avg_participation,
            'avg_training_time': avg_training_time
        }), 200
    except Exception as e:
        logger.error(f"Error getting client stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@client_bp.route('/performance', methods=['GET'])
def get_client_performance():
    """获取客户端性能监控数据"""
    try:
        current_clients = get_current_clients()
        
        # 1. 参与度分布 (Participation Distribution)
        participation_data = []
        for client_id, client in current_clients.items():
            participation_data.append({
                'name': f'Client {client_id}',
                'value': client['participation_count']
            })
            
        # 2. 训练时间统计 (Training Time Statistics)
        training_time_data = []
        for client_id, client in current_clients.items():
            training_time_data.append({
                'name': f'Client {client_id}',
                'value': float(client['avg_training_time'])
            })
            
        # 3. 系统资源使用情况 (System Resource Usage)
        # 模拟根据当前活跃客户端数量产生的资源占用
        num_active = sum(1 for c in current_clients.values() if c.get('status') == 'active')
        num_busy = sum(1 for c in current_clients.values() if c.get('status') == 'busy')
        
        # 基础占用 + 动态占用
        cpu_usage = round(psutil.cpu_percent(interval=0.1), 1)
        mem_usage = round(psutil.virtual_memory().percent, 1)
        network_latency = max(10, 30 + num_busy * 5 + np.random.randint(-10, 10))
        bandwidth = max(10, 100 - num_busy * 10 + np.random.randint(-5, 5))
        gpu_resources = get_gpu_resources()
        
        return jsonify({
            'participation_distribution': participation_data,
            'training_time_stats': training_time_data,
            'system_resources': {
                'cpu': cpu_usage,
                'memory': mem_usage,
                'gpu': gpu_resources,
                'gpu_usage': gpu_resources['usage'],
                'gpu_memory': gpu_resources['memory'],
                'latency': network_latency,
                'bandwidth': bandwidth
            }
        }), 200
    except Exception as e:
        logger.error(f"Error getting client performance: {str(e)}")
        return jsonify({'error': str(e)}), 500
