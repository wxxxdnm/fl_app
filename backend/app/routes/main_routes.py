from flask import Blueprint, jsonify, send_from_directory
from . import train_routes, client_routes, data_routes
from ..services.activity_service import activity_service
from ..services.history_service import history_service
import os

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """返回API信息"""
    return jsonify({
        'message': '联邦学习平台API',
        'version': '1.0.0',
        'endpoints': {
            'datasets': '/api/data/datasets',
            'train': '/api/train/start',
            'status': '/api/train/status',
            'clients': '/api/clients/stats',
            'visualization': '/api/viz/training_curves',
            'dashboard': '/api/main/dashboard_stats'
        }
    })

@main_bp.route('/api/main/dashboard_stats')
def get_dashboard_stats():
    """获取主页仪表盘统计数据"""
    try:
        # 1. 获取客户端统计
        # 优先从 fl_system 获取参与训练的客户端，如果没有则从 client_routes 获取注册的客户端
        total_clients = 0
        if train_routes.fl_system is not None:
            total_clients = len(train_routes.fl_system.clients)
        
        if total_clients == 0:
            total_clients = len(client_routes.static_clients)
        
        # 2. 获取训练历史和最新准确率
        latest_accuracy = 0
        total_rounds = 0
        training_history = []
        
        if train_routes.fl_system is not None:
            history = train_routes.fl_system.get_training_history()
            if history:
                latest_accuracy = history[-1]['global_metrics']['accuracy'] * 100
                total_rounds = len(history)
                for h in history:
                    training_history.append({
                        'round': h['round'],
                        'accuracy': h['global_metrics']['accuracy'],
                        'loss': h['global_metrics']['loss']
                    })
        if not training_history:
            latest_run = history_service.get_latest_training_run()
            if latest_run:
                latest_accuracy = latest_run.get('final_accuracy', 0) * 100
                total_rounds = latest_run.get('rounds', 0)
                total_clients = total_clients or latest_run.get('num_clients', 0)
                training_history = [
                    {
                        'round': h['round'],
                        'accuracy': h['global_metrics']['accuracy'],
                        'loss': h['global_metrics']['loss']
                    }
                    for h in latest_run.get('history', [])
                    if 'global_metrics' in h
                ]

        num_datasets = len(data_routes.AVAILABLE_DATASETS)
        training_runs = history_service.get_training_runs(10)
        saved_models = history_service.get_model_records(10)
        recent_activities = activity_service.get_activities()
        
        return jsonify({
            'total_clients': total_clients,
            'latest_accuracy': latest_accuracy,
            'total_rounds': total_rounds,
            'num_datasets': num_datasets,
            'training_history': training_history,
            'training_runs': training_runs,
            'saved_models': saved_models,
            'activities': recent_activities
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/health')
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'timestamp': __import__('datetime').datetime.now().isoformat()})
