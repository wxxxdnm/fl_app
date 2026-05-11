from flask import Blueprint, jsonify, request
from ..services.federated_learning import FederatedLearning, FLClient
from ..services.model_manager import ModelManager
from ..services.data_manager import DataManager
from ..services.activity_service import activity_service
from ..services.history_service import history_service
from ..services.visualization_service import get_visualization_snapshot
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
current_training_config = {}
training_stop_requested = False
active_training_id = 0
training_thread = None

def get_training_activity_subject(training_config):
    dataset_name = str(training_config.get('dataset_name') or 'unknown').upper()
    model_name = training_config.get('model_name') or 'model'
    aggregation_algorithm = training_config.get('aggregation_algorithm') or 'fedavg'
    return f"{dataset_name} / {model_name} / {aggregation_algorithm}"

def build_training_activity_metadata(training_id, training_config, event, status=None):
    return {
        'category': 'training',
        'training': {
            'id': training_id,
            'event': event,
            'status': status,
            'dataset_name': training_config.get('dataset_name'),
            'model_name': training_config.get('model_name'),
            'aggregation_algorithm': training_config.get('aggregation_algorithm'),
            'num_clients': training_config.get('num_clients'),
            'num_rounds': training_config.get('num_rounds'),
            'client_fraction': training_config.get('client_fraction'),
            'iid': training_config.get('iid')
        }
    }

def add_training_activity(content, activity_type, training_id, training_config, event, status=None):
    activity_service.add_activity(
        content,
        activity_type,
        build_training_activity_metadata(training_id, training_config, event, status)
    )

@train_bp.route('/algorithms', methods=['GET'])
def get_aggregation_algorithms():
    """获取支持的联邦聚合算法"""
    return jsonify({'algorithms': FederatedLearning.get_available_algorithms()}), 200

def run_training_loop(num_rounds, client_fraction, num_clients, training_id, training_system, training_config):
    """后台训练循环"""
    global training_status, training_history, training_error
    try:
        training_status = "Running"
        training_subject = get_training_activity_subject(training_config)
        add_training_activity(
            f"开始训练模型：{training_subject}",
            "process",
            training_id,
            training_config,
            "start",
            "running"
        )
        
        for round_num in range(num_rounds):
            if training_stop_requested or training_id != active_training_id:
                if training_id == active_training_id:
                    training_status = "Stopped"
                    add_training_activity(
                        f"模型训练结束：{training_subject}（已停止）",
                        "warning",
                        training_id,
                        training_config,
                        "end",
                        "stopped"
                    )
                return

            # 选择部分客户端参与本轮训练
            num_selected = max(1, int(client_fraction * num_clients))
            indices = torch.randperm(num_clients)[:num_selected].tolist()
            selected_clients = [training_system.clients[i] for i in indices]
            
            # 设置选中的客户端状态为忙碌
            for client in selected_clients:
                client.status = 'busy'

            # 执行一轮训练
            round_result = training_system.train_round(selected_clients)
            if training_id != active_training_id:
                return
            training_history.append(round_result)
            
            # 恢复客户端状态为活跃
            for client in selected_clients:
                client.status = 'active'

            logger.info(f"Round {round_num + 1}/{num_rounds} completed")
            
            if training_stop_requested:
                training_status = "Stopped"
                add_training_activity(
                    f"模型训练结束：{training_subject}（已停止）",
                    "warning",
                    training_id,
                    training_config,
                    "end",
                    "stopped"
                )
                return

        # 评估最终模型
        if training_stop_requested or training_id != active_training_id:
            if training_id == active_training_id:
                training_status = "Stopped"
                add_training_activity(
                    f"模型训练结束：{training_subject}（已停止）",
                    "warning",
                    training_id,
                    training_config,
                    "end",
                    "stopped"
                )
            return
        training_system.evaluate_global_model(training_system.clients)
        training_status = "Completed"
        visualization_snapshot = {}
        try:
            visualization_snapshot = get_visualization_snapshot(training_system, training_config.get('dataset_name'))
        except Exception as snapshot_error:
            logger.warning(f"Failed to build visualization snapshot: {snapshot_error}")
        history_service.add_training_run(
            training_config,
            training_history,
            training_status,
            visualization=visualization_snapshot
        )
        final_accuracy = training_history[-1]['global_metrics']['accuracy'] * 100 if training_history else 0
        add_training_activity(
            f"模型训练完成：{training_subject}，最终准确率: {final_accuracy:.2f}%",
            "success",
            training_id,
            training_config,
            "end",
            "completed"
        )

    except Exception as e:
        if training_id != active_training_id:
            return
        training_status = "Error"
        training_error = str(e)
        history_service.add_training_run(training_config, training_history, training_status, training_error)
        logger.error(f"Training loop error: {training_error}")
        training_subject = get_training_activity_subject(training_config)
        add_training_activity(
            f"模型训练结束：{training_subject}（失败：{training_error}）",
            "error",
            training_id,
            training_config,
            "end",
            "error"
        )

@train_bp.route('/start', methods=['POST'])
def start_training():
    """启动联邦学习训练"""
    global fl_system, training_history, training_status, training_error, current_training_config
    global training_stop_requested, active_training_id, training_thread
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
            server_lr_value = data.get('server_lr')
            server_lr = float(server_lr_value) if server_lr_value is not None else None
            server_momentum = float(data.get('server_momentum', 0.9))
            proximal_mu = float(data.get('proximal_mu', 0.01))
            adaptive_beta1 = float(data.get('adaptive_beta1', 0.9))
            adaptive_beta2 = float(data.get('adaptive_beta2', 0.99))
            adaptive_tau = float(data.get('adaptive_tau', 1e-3))
            non_iid_classes_per_client = int(data.get('non_iid_classes_per_client', 2))
            non_iid_seed = int(data.get('non_iid_seed', 42))
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
        if server_lr is not None and server_lr <= 0:
            return jsonify({'error': 'server_lr must be greater than 0'}), 400
        if not 0 <= server_momentum < 1:
            return jsonify({'error': 'server_momentum must be in [0, 1)'}), 400
        if proximal_mu < 0:
            return jsonify({'error': 'proximal_mu must be non-negative'}), 400
        if not 0 <= adaptive_beta1 < 1 or not 0 <= adaptive_beta2 < 1:
            return jsonify({'error': 'adaptive_beta1 and adaptive_beta2 must be in [0, 1)'}), 400
        if adaptive_tau <= 0:
            return jsonify({'error': 'adaptive_tau must be greater than 0'}), 400
        if non_iid_classes_per_client < 1:
            return jsonify({'error': 'non_iid_classes_per_client must be at least 1'}), 400
        model_manager = ModelManager()
        try:
            model_config = model_manager.get_model_config(dataset_name)
        except ValueError as error:
            return jsonify({'error': str(error)}), 400
        supported_model_names = {model['value'] for model in model_config['models']}
        if model_name and model_name not in supported_model_names:
            supported = ', '.join(sorted(supported_model_names))
            return jsonify({'error': f'Unsupported model for {dataset_name}: {model_name}. Supported: {supported}'}), 400
        if not iid:
            num_classes = model_config['num_classes']
            if num_clients * non_iid_classes_per_client < num_classes:
                return jsonify({
                    'error': (
                        'num_clients * non_iid_classes_per_client must cover all dataset classes '
                        f'({num_clients} * {non_iid_classes_per_client} < {num_classes})'
                    )
                }), 400
        if aggregation_algorithm not in FederatedLearning.SUPPORTED_ALGORITHMS:
            supported = ', '.join(sorted(FederatedLearning.SUPPORTED_ALGORITHMS))
            return jsonify({'error': f'Unsupported aggregation_algorithm. Supported: {supported}'}), 400
        if (
            aggregation_algorithm in FederatedLearning.ADAPTIVE_ALGORITHMS
            and server_lr is not None
            and server_lr > FederatedLearning.MAX_ADAPTIVE_SERVER_LR
        ):
            return jsonify({
                'error': (
                    f'server_lr for {FederatedLearning.SUPPORTED_ALGORITHMS[aggregation_algorithm]} '
                    f'must be <= {FederatedLearning.MAX_ADAPTIVE_SERVER_LR}'
                )
            }), 400
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU for training.")
            device = 'cpu'

        # 初始化联邦学习系统
        global_model = model_manager.create_model(dataset_name, model_name)
        selected_model_name = getattr(global_model, 'model_name', model_name)
        current_training_config = {
            'dataset_name': dataset_name,
            'model_name': selected_model_name,
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'client_fraction': client_fraction,
            'batch_size': batch_size,
            'aggregation_algorithm': aggregation_algorithm,
            'iid': iid,
            'non_iid_classes_per_client': non_iid_classes_per_client,
            'non_iid_seed': non_iid_seed,
            'device': device
        }

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
        fl_system.model_name = selected_model_name
        current_training_config.update({
            'server_lr': fl_system.server_lr,
            'server_momentum': server_momentum,
            'proximal_mu': proximal_mu,
            'adaptive_beta1': adaptive_beta1,
            'adaptive_beta2': adaptive_beta2,
            'adaptive_tau': adaptive_tau
        })

        # 准备数据
        data_manager = DataManager()
        client_dataloaders = data_manager.create_federated_datasets(
            dataset_name=dataset_name,
            num_clients=num_clients,
            batch_size=batch_size,
            iid=iid,
            non_iid_classes_per_client=non_iid_classes_per_client,
            non_iid_seed=non_iid_seed
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
        training_stop_requested = False
        active_training_id += 1
        current_training_id = active_training_id
        training_status = "Running"
        
        # 启动后台线程执行训练
        training_thread = threading.Thread(
            target=run_training_loop,
            args=(num_rounds, client_fraction, num_clients, current_training_id, fl_system, current_training_config.copy()),
            daemon=True
        )
        training_thread.start()

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
        if training_status == "Stopped":
            history = fl_system.get_training_history() if fl_system is not None else training_history
            response = {
                'status': training_status,
                'current_round': len(history),
                'history': history
            }
            if history:
                response['latest_metrics'] = history[-1]['global_metrics']
            return jsonify(response), 200

        if fl_system is None:
            latest_run = history_service.get_latest_training_run()
            if latest_run:
                return jsonify({
                    'status': latest_run.get('status', 'Completed'),
                    'current_round': latest_run.get('rounds', 0),
                    'history': latest_run.get('history', []),
                    'latest_metrics': latest_run.get('history', [{}])[-1].get('global_metrics', {}) if latest_run.get('history') else {},
                    'aggregation_algorithm': latest_run.get('aggregation_algorithm')
                }), 200
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

@train_bp.route('/history/<run_id>', methods=['DELETE'])
def delete_training_history(run_id):
    try:
        deleted = history_service.delete_training_run(run_id)
        if not deleted:
            return jsonify({'error': 'Training history record not found'}), 404
        activity_service.add_activity("已删除一条历史训练记录", "info")
        return jsonify({'message': 'Training history record deleted'}), 200
    except Exception as e:
        logger.error(f"Error deleting training history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@train_bp.route('/stop', methods=['POST'])
def stop_training():
    """停止训练"""
    try:
        global training_status, training_stop_requested, active_training_id
        training_stop_requested = True
        active_training_id += 1
        training_status = "Stopped"
        return jsonify({'message': 'Training stopped', 'status': training_status}), 200
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
        history_service.add_model_record(path, {
            'dataset_name': getattr(fl_system, 'dataset_name', None),
            'model_name': getattr(fl_system, 'model_name', None),
            'model_class': fl_system.global_model.__class__.__name__,
            'rounds': len(fl_system.get_training_history()),
            'num_clients': len(fl_system.clients),
            'aggregation_algorithm': fl_system.aggregation_algorithm
        })

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
