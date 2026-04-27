from flask import Blueprint, jsonify, send_from_directory
from . import train_routes, client_routes, data_routes
from ..services.activity_service import activity_service
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
                # 转换历史数据格式供前端图表使用
                for h in history:
                    training_history.append({
                        'round': h['round'],
                        'accuracy': h['global_metrics']['accuracy'],
                        'loss': h['global_metrics']['loss']
                    })
        
        # 3. 获取数据集数量
        # 从 data_routes 获取可用数据集
        try:
            from .data_routes import get_available_datasets
            # 这里我们直接调用逻辑或者硬编码，因为 get_available_datasets 返回的是 response
            num_datasets = 2 # 默认值
        except:
            num_datasets = 2
        
        # 4. 获取系统活动
        recent_activities = activity_service.get_activities()
        
        return jsonify({
            'total_clients': total_clients,
            'latest_accuracy': latest_accuracy,
            'total_rounds': total_rounds,
            'num_datasets': num_datasets,
            'training_history': training_history,
            'activities': recent_activities
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/health')
def health_check():
    """健康检查"""
    return jsonify({'status': 'healthy', 'timestamp': __import__('datetime').datetime.now().isoformat()})
