from flask import Blueprint, jsonify, request
from ..services.model_manager import ModelManager
from ..services.activity_service import activity_service
from ..services.history_service import history_service
from .utils import get_json_body
import logging
import os

logger = logging.getLogger(__name__)
model_bp = Blueprint('model', __name__)
model_manager = ModelManager()

@model_bp.route('/models', methods=['GET'])
def get_available_models():
    """获取可用的模型列表"""
    try:
        dataset_name = request.args.get('dataset_name')
        models = model_manager.get_available_models(dataset_name)
        return jsonify({'models': models}), 200
    except ValueError as e:
        logger.exception(f"Error getting models: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Error getting models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@model_bp.route('/models/<dataset_name>/create', methods=['POST'])
def create_model(dataset_name):
    """为指定数据集创建模型"""
    try:
        data = request.get_json(silent=True) or {}
        model_name = data.get('model_name')
        model = model_manager.create_model(dataset_name, model_name)
        summary = model_manager.get_model_summary(model)

        activity_service.add_activity(f"为数据集 {dataset_name} 创建了新模型", "info")
        return jsonify({
            'message': f'Model created for {dataset_name}',
            'model_info': summary
        }), 200
    except ValueError as e:
        logger.exception(f"Error creating model: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Error creating model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@model_bp.route('/models/<dataset_name>/config', methods=['GET'])
def get_model_config(dataset_name):
    """获取模型配置信息"""
    try:
        config = model_manager.get_model_config(dataset_name)
        return jsonify(config), 200
    except ValueError as e:
        logger.exception(f"Error getting model config: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Error getting model config: {str(e)}")
        return jsonify({'error': str(e)}), 500

@model_bp.route('/history', methods=['GET'])
def get_model_history():
    try:
        try:
            limit = int(request.args.get('limit', 50))
        except (TypeError, ValueError):
            return jsonify({'error': 'limit must be an integer'}), 400
        if limit < 1:
            return jsonify({'error': 'limit must be at least 1'}), 400
        return jsonify({'models': history_service.get_model_records(limit)}), 200
    except Exception as e:
        logger.exception(f"Error getting model history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@model_bp.route('/models/save', methods=['POST'])
def save_model():
    """保存模型"""
    try:
        data, error_response = get_json_body()
        if error_response:
            return error_response
        dataset_name = data.get('dataset_name')
        model_name = data.get('model_name')
        path = data.get('path', './checkpoints/model.pth')
        metadata = data.get('metadata', {})
        if not dataset_name:
            return jsonify({'error': 'dataset_name is required to create and save a model'}), 400

        model = model_manager.create_model(dataset_name, model_name)
        model_manager.save_model(model, path, metadata)
        history_service.add_model_record(path, {
            'dataset_name': dataset_name,
            'model_name': getattr(model, 'model_name', model_name),
            'model_class': model.__class__.__name__,
            **metadata
        })
        activity_service.add_activity(f"模型已保存至 {path}", "success")
        return jsonify({'message': 'Model saved successfully', 'path': path}), 200
    except ValueError as e:
        logger.exception(f"Error saving model: {str(e)}")
        activity_service.add_activity(f"保存模型失败: {str(e)}", "error")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Error saving model: {str(e)}")
        activity_service.add_activity(f"保存模型失败: {str(e)}", "error")
        return jsonify({'error': str(e)}), 500

@model_bp.route('/models/load', methods=['POST'])
def load_model():
    """加载模型"""
    try:
        data, error_response = get_json_body()
        if error_response:
            return error_response
        path = data.get('path', './checkpoints/model.pth')
        if not os.path.isfile(path):
            return jsonify({'error': 'Model file not found'}), 404

        model = model_manager.load_model(path)

        activity_service.add_activity(f"已加载模型: {path}", "info")
        return jsonify({
            'message': 'Model loaded successfully',
            'model_class': model.__class__.__name__
        }), 200
    except ValueError as e:
        logger.exception(f"Error loading model: {str(e)}")
        activity_service.add_activity(f"加载模型失败: {str(e)}", "error")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Error loading model: {str(e)}")
        activity_service.add_activity(f"加载模型失败: {str(e)}", "error")
        return jsonify({'error': str(e)}), 500
