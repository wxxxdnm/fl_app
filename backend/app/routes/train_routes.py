from flask import Blueprint, jsonify, request
from ..services.federated_learning import FederatedLearning, FLClient
from ..services.model_manager import ModelManager
from ..services.data_manager import DataManager
from ..services.activity_service import activity_service
from .utils import get_json_body
import torch
import logging

import threading

logger = logging.getLogger(__name__)
train_bp = Blueprint('train', __name__)

# 全局变量存储训练状态
fl_system = None
training_history = []
training_status = "Not started"  # "Not started", "Running", "Completed", "Error"
training_error = None

@train_bp.route('/algorithms', methods=['GET'])
def get_aggregation_algorithms():
    """获取支持的联邦聚合算法"""
    return jsonify({'algorithms': FederatedLearning.get_available_algorithms()}), 200

def run_training_loop(num_rounds, client_fraction, num_clients):
    """后台训练循环"""
    global training_status, training_history, training_error
    try:
        training_status = "Running"
        activity_service.add_activity(
            f"联邦学习训练开始：算法 {fl_system.aggregation_algorithm}，总共 {num_rounds} 轮，参与比例 {client_fraction}",
            "process"
        )
        
        for round_num in range(num_rounds):
            if fl_system is None:  # 训练被停止
                activity_service.add_activity("训练已手动停止", "warning")
                return

            # 选择部分客户端参与本轮训练
            num_selected = max(1, int(client_fraction * num_clients))
            indices = torch.randperm(num_clients)[:num_selected].tolist()
            selected_clients = [fl_system.clients[i] for i in indices]
            
            # 设置选中的客户端状态为忙碌
            for client in selected_clients:
                client.status = 'busy'

            # 执行一轮训练
            round_result = fl_system.train_round(selected_clients)
            training_history.append(round_result)
            
            # 恢复客户端状态为活跃
            for client in selected_clients:
                client.status = 'active'

            logger.info(f"Round {round_num + 1}/{num_rounds} completed")
            
            if (round_num + 1) % 5 == 0 or round_num == 0:
                activity_service.add_activity(f"第 {round_num + 1} 轮训练完成，准确率: {round_result['global_metrics']['accuracy']*100:.2f}%", "success")

        # 评估最终模型
        fl_system.evaluate_global_model(fl_system.clients)
        training_status = "Completed"
        activity_service.add_activity(f"联邦学习训练圆满完成！最终准确率: {training_history[-1]['global_metrics']['accuracy']*100:.2f}%", "success")

    except Exception as e:
        training_status = "Error"
        training_error = str(e)
        logger.error(f"Training loop error: {training_error}")
        activity_service.add_activity(f"训练过程中出现错误: {training_error}", "error")

@train_bp.route('/start', methods=['POST'])
def start_training():
    """启动联邦学习训练"""
    global fl_system, training_history, training_status, training_error
    try:
        if training_status == "Running":
            return jsonify({'error': 'Training is already running'}), 400

        data, error_response = get_json_body()
        if error_response:
            return error_response
        dataset_name = data.get('dataset_name', 'mnist')
        model_name = data.get('model_name')
        try:
            num_clients = int(data.get('num_clients', 10))
            num_rounds = int(data.get('num_rounds', 10))
            client_fraction = float(data.get('client_fraction', 0.5))
            batch_size = int(data.get('batch_size', 64))
            server_lr = float(data.get('server_lr', 1.0))
            server_momentum = float(data.get('server_momentum', 0.9))
            proximal_mu = float(data.get('proximal_mu', 0.01))
            adaptive_beta1 = float(data.get('adaptive_beta1', 0.9))
            adaptive_beta2 = float(data.get('adaptive_beta2', 0.99))
            adaptive_tau = float(data.get('adaptive_tau', 1e-3))
        except (TypeError, ValueError):
            return jsonify({'error': 'Training numeric parameters are invalid'}), 400
        aggregation_algorithm = data.get('aggregation_algorithm', 'fedavg').lower()
        iid = data.get('iid', True)
        device = data.get('device', 'cuda')
        if num_clients < 1:
            return jsonify({'error': 'num_clients must be at least 1'}), 400
        if num_rounds < 1:
            return jsonify({'error': 'num_rounds must be at least 1'}), 400
        if not 0 < client_fraction <= 1:
            return jsonify({'error': 'client_fraction must be between 0 and 1'}), 400
        if batch_size < 1:
            return jsonify({'error': 'batch_size must be at least 1'}), 400
        if server_lr <= 0:
            return jsonify({'error': 'server_lr must be greater than 0'}), 400
        if not 0 <= server_momentum < 1:
            return jsonify({'error': 'server_momentum must be in [0, 1)'}), 400
        if proximal_mu < 0:
            return jsonify({'error': 'proximal_mu must be non-negative'}), 400
        if not 0 <= adaptive_beta1 < 1 or not 0 <= adaptive_beta2 < 1:
            return jsonify({'error': 'adaptive_beta1 and adaptive_beta2 must be in [0, 1)'}), 400
        if adaptive_tau <= 0:
            return jsonify({'error': 'adaptive_tau must be greater than 0'}), 400
        if aggregation_algorithm not in FederatedLearning.SUPPORTED_ALGORITHMS:
            supported = ', '.join(sorted(FederatedLearning.SUPPORTED_ALGORITHMS))
            return jsonify({'error': f'Unsupported aggregation_algorithm. Supported: {supported}'}), 400
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU for training.")
            device = 'cpu'

        # 初始化联邦学习系统
        model_manager = ModelManager()
        global_model = model_manager.create_model(dataset_name, model_name)

        fl_system = FederatedLearning(
            global_model,
            device,
            aggregation_algorithm=aggregation_algorithm,
            server_lr=server_lr,
            server_momentum=server_momentum,
            proximal_mu=proximal_mu,
            adaptive_beta1=adaptive_beta1,
            adaptive_beta2=adaptive_beta2,
            adaptive_tau=adaptive_tau
        )
        fl_system.dataset_name = dataset_name
        fl_system.model_name = getattr(global_model, 'model_name', model_name)

        # 准备数据
        data_manager = DataManager()
        client_dataloaders = data_manager.create_federated_datasets(
            dataset_name=dataset_name,
            num_clients=num_clients,
            batch_size=batch_size,
            iid=iid
        )

        # 创建客户端
        for i in range(num_clients):
            train_loader = client_dataloaders[f'client_{i}_train']
            test_loader = client_dataloaders[f'client_{i}_test']
            client = FLClient(i, global_model, train_loader, test_loader, device)
            fl_system.add_client(client)

        # 开始训练
        training_history = []
        training_error = None
        
        # 启动后台线程执行训练
        thread = threading.Thread(
            target=run_training_loop,
            args=(num_rounds, client_fraction, num_clients),
            daemon=True
        )
        thread.start()

        return jsonify({
            'message': 'Training started in background',
            'status': 'Running'
        }), 200

    except Exception as e:
        training_status = "Error"
        training_error = str(e)
        logger.error(f"Error starting training: {training_error}")
        return jsonify({'error': training_error}), 500

@train_bp.route('/status', methods=['GET'])
def get_training_status():
    """获取训练状态"""
    try:
        if fl_system is None:
            return jsonify({'status': 'Not started'}), 200

        history = fl_system.get_training_history()
        
        response = {
            'status': training_status,
            'current_round': len(history),
            'history': history,
            'aggregation_algorithm': fl_system.aggregation_algorithm
        }

        if history:
            response['latest_metrics'] = history[-1]['global_metrics']
        
        if training_status == "Error":
            response['error'] = training_error

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@train_bp.route('/stop', methods=['POST'])
def stop_training():
    """停止训练"""
    try:
        global fl_system
        fl_system = None
        return jsonify({'message': 'Training stopped'}), 200
    except Exception as e:
        logger.error(f"Error stopping training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@train_bp.route('/save', methods=['POST'])
def save_trained_model():
    """保存训练好的模型"""
    try:
        if fl_system is None:
            return jsonify({'error': 'No trained model available'}), 400

        data, error_response = get_json_body()
        if error_response:
            return error_response
        path = data.get('path', './checkpoints/federated_model.pth')

        fl_system.save_model(path)

        return jsonify({'message': f'Model saved to {path}'}), 200
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@train_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """获取训练指标"""
    try:
        if fl_system is None:
            return jsonify({'error': 'No training data available'}), 400

        history = fl_system.get_training_history()
        metrics_data = {
            'rounds': [],
            'accuracies': [],
            'losses': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': [],
            'balanced_accuracies': [],
            'samples_per_second': [],
            'client_accuracies': [],
            'client_losses': [],
            'client_precisions': [],
            'client_recalls': [],
            'client_f1_scores': []
        }

        for round_data in history:
            global_metrics = round_data['global_metrics']
            metrics_data['rounds'].append(round_data['round'])
            metrics_data['accuracies'].append(global_metrics['accuracy'])
            metrics_data['losses'].append(global_metrics['loss'])
            metrics_data['precisions'].append(global_metrics.get('precision', 0))
            metrics_data['recalls'].append(global_metrics.get('recall', 0))
            metrics_data['f1_scores'].append(global_metrics.get('f1_score', 0))
            metrics_data['balanced_accuracies'].append(global_metrics.get('balanced_accuracy', 0))
            metrics_data['samples_per_second'].append(global_metrics.get('samples_per_second', 0))

            # 客户端指标
            client_accs = [m['accuracy'] for m in round_data['client_metrics']]
            client_losses = [m['loss'] for m in round_data['client_metrics']]
            client_precisions = [m.get('precision', 0) for m in round_data['client_metrics']]
            client_recalls = [m.get('recall', 0) for m in round_data['client_metrics']]
            client_f1_scores = [m.get('f1_score', 0) for m in round_data['client_metrics']]
            metrics_data['client_accuracies'].append(client_accs)
            metrics_data['client_losses'].append(client_losses)
            metrics_data['client_precisions'].append(client_precisions)
            metrics_data['client_recalls'].append(client_recalls)
            metrics_data['client_f1_scores'].append(client_f1_scores)

        return jsonify(metrics_data), 200

    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500
