from flask import Blueprint, jsonify, request
from . import train_routes
from .utils import get_json_body
from ..services.data_manager import DataManager
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
import time
from torch.utils.data import Subset
from ..services.federated_learning import calculate_classification_metrics, update_confusion_matrix

logger = logging.getLogger(__name__)
viz_bp = Blueprint('visualization', __name__)

def get_dataset_class_names(dataset_name):
    try:
        info = DataManager().get_dataset_info(dataset_name)
        return [str(class_name) for class_name in info['classes']]
    except Exception:
        return [str(i) for i in range(10)]

def get_current_dataset_name():
    if train_routes.fl_system is not None:
        dataset_name = getattr(train_routes.fl_system, 'dataset_name', None)
        if dataset_name:
            return dataset_name
    return train_routes.current_training_config.get('dataset_name') or 'mnist'

def extract_labels(dataset):
    """Extract labels from regular datasets or nested Subset instances."""
    if isinstance(dataset, Subset):
        parent_labels = extract_labels(dataset.dataset)
        return [parent_labels[int(index)] for index in dataset.indices]

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

    labels = []
    for _, target in dataset:
        if torch.is_tensor(target):
            target = target.item()
        labels.append(int(target))
    return labels

def evaluate_global_model_on_client(fl_system, client):
    model = fl_system.global_model
    model.eval()
    start_time = time.time()
    total_loss = 0
    correct = 0
    total = 0
    num_classes = getattr(model, 'num_classes', None)
    confusion_matrix = None

    with torch.no_grad():
        for data, target in client.test_loader:
            data, target = data.to(fl_system.device), target.to(fl_system.device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if num_classes is None:
                num_classes = output.size(1)
            confusion_matrix = update_confusion_matrix(confusion_matrix, target, predicted, num_classes)

    evaluation_time = time.time() - start_time
    metrics = {
        'loss': total_loss / max(1, len(client.test_loader)),
        'accuracy': correct / max(1, total),
        'num_samples': total,
        'evaluation_time': evaluation_time,
        'samples_per_second': total / evaluation_time if evaluation_time > 0 else 0.0
    }
    if confusion_matrix is not None:
        metrics.update(calculate_classification_metrics(confusion_matrix))
    return metrics

def compute_distribution_stats(count_matrix):
    if count_matrix.size == 0 or count_matrix.sum() == 0:
        return {
            'class_balance': 0,
            'data_quality': 0,
            'sample_quantity': 0,
            'feature_diversity': 0,
            'data_consistency': 0
        }

    client_totals = count_matrix.sum(axis=1)
    non_empty = client_totals > 0
    non_empty_counts = count_matrix[non_empty]
    non_empty_totals = client_totals[non_empty]
    num_classes = count_matrix.shape[1]

    distributions = non_empty_counts / non_empty_totals[:, np.newaxis]
    entropy = -np.sum(np.where(distributions > 0, distributions * np.log(distributions + 1e-12), 0), axis=1)
    class_balance = float(np.mean(entropy / np.log(max(num_classes, 2))) * 100)

    expected_total = np.mean(non_empty_totals)
    sample_cv = np.std(non_empty_totals) / expected_total if expected_total else 0
    sample_quantity = float(max(0, min(100, 100 * (1 - sample_cv))))

    global_distribution = count_matrix.sum(axis=0)
    covered_classes = np.count_nonzero(global_distribution)
    feature_diversity = float((covered_classes / num_classes) * 100) if num_classes else 0

    client_class_presence = np.count_nonzero(non_empty_counts, axis=1)
    data_quality = float(np.mean(client_class_presence / num_classes) * 100) if num_classes else 0

    global_dist = global_distribution / global_distribution.sum()
    mean_abs_deviation = np.mean(np.abs(distributions - global_dist))
    data_consistency = float(max(0, min(100, 100 * (1 - mean_abs_deviation * num_classes / 2))))

    return {
        'class_balance': round(class_balance, 1),
        'data_quality': round(data_quality, 1),
        'sample_quantity': round(sample_quantity, 1),
        'feature_diversity': round(feature_diversity, 1),
        'data_consistency': round(data_consistency, 1)
    }

@viz_bp.route('/training_curves', methods=['GET'])
def get_training_curves():
    """获取训练曲线数据"""
    try:
        if train_routes.fl_system is None:
            return jsonify({'error': 'No training data available'}), 400

        history = train_routes.fl_system.get_training_history()
        if not history:
            return jsonify({'error': 'No training history available'}), 400

        # 提取数据
        rounds = [h['round'] for h in history]
        global_accuracies = [h['global_metrics']['accuracy'] for h in history]
        global_losses = [h['global_metrics']['loss'] for h in history]
        global_precisions = [h['global_metrics'].get('precision', 0) for h in history]
        global_recalls = [h['global_metrics'].get('recall', 0) for h in history]
        global_f1_scores = [h['global_metrics'].get('f1_score', 0) for h in history]
        global_balanced_accuracies = [h['global_metrics'].get('balanced_accuracy', 0) for h in history]
        global_samples_per_second = [h['global_metrics'].get('samples_per_second', 0) for h in history]

        # 客户端数据
        client_accuracies = []
        client_losses = []
        for h in history:
            client_accs = [m['accuracy'] for m in h['client_metrics']]
            client_losses_list = [m['loss'] for m in h['client_metrics']]
            client_accuracies.append(client_accs)
            client_losses.append(client_losses_list)

        return jsonify({
            'rounds': rounds,
            'global_accuracies': global_accuracies,
            'global_losses': global_losses,
            'global_precisions': global_precisions,
            'global_recalls': global_recalls,
            'global_f1_scores': global_f1_scores,
            'global_balanced_accuracies': global_balanced_accuracies,
            'global_samples_per_second': global_samples_per_second,
            'client_accuracies': client_accuracies,
            'client_losses': client_losses
        }), 200

    except Exception as e:
        logger.exception(f"Error getting training curves: {str(e)}")
        return jsonify({'error': str(e)}), 500

@viz_bp.route('/model_performance', methods=['GET'])
def get_model_performance():
    """获取模型性能可视化数据"""
    try:
        if train_routes.fl_system is None:
            return jsonify({'error': 'No model available'}), 400

        # 评估所有客户端
        client_performance = []
        for client in train_routes.fl_system.clients:
            metrics = evaluate_global_model_on_client(train_routes.fl_system, client)
            client_performance.append({
                'client_id': client.client_id,
                'accuracy': metrics['accuracy'],
                'loss': metrics['loss'],
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                'samples_per_second': metrics.get('samples_per_second', 0),
                'num_samples': metrics['num_samples']
            })

        # 创建性能对比数据
        df = pd.DataFrame(client_performance)
        performance_data = {
            'client_ids': df['client_id'].tolist(),
            'accuracies': df['accuracy'].tolist(),
            'losses': df['loss'].tolist(),
            'precisions': df['precision'].tolist(),
            'recalls': df['recall'].tolist(),
            'f1_scores': df['f1_score'].tolist(),
            'balanced_accuracies': df['balanced_accuracy'].tolist(),
            'samples_per_second': df['samples_per_second'].tolist(),
            'sample_sizes': df['num_samples'].tolist(),
            'stats': {
                'mean_accuracy': float(df['accuracy'].mean()),
                'std_accuracy': float(df['accuracy'].std()),
                'mean_loss': float(df['loss'].mean()),
                'std_loss': float(df['loss'].std()),
                'mean_precision': float(df['precision'].mean()),
                'mean_recall': float(df['recall'].mean()),
                'mean_f1_score': float(df['f1_score'].mean()),
                'mean_balanced_accuracy': float(df['balanced_accuracy'].mean())
            }
        }

        return jsonify(performance_data), 200

    except Exception as e:
        logger.exception(f"Error getting model performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@viz_bp.route('/confusion_matrix', methods=['POST'])
def get_confusion_matrix():
    """生成混淆矩阵数据"""
    try:
        data, error_response = get_json_body()
        if error_response:
            return error_response
        dataset_name = data.get('dataset_name') or get_current_dataset_name()

        class_names = get_dataset_class_names(dataset_name)
        num_classes = len(class_names)

        # 生成模拟混淆矩阵
        if train_routes.fl_system is None or not train_routes.fl_system.clients:
            return jsonify({'error': 'No model available'}), 400

        model = train_routes.fl_system.global_model
        device = train_routes.fl_system.device
        model.eval()
        confusion_counts = np.zeros((num_classes, num_classes), dtype=np.int64)

        # 对角线设置更高的值（正确预测）
        with torch.no_grad():
            for client in train_routes.fl_system.clients:
                for inputs, targets in client.test_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    predictions = outputs.argmax(dim=1).cpu().numpy()
                    actuals = targets.cpu().numpy()
                    for actual, predicted in zip(actuals, predictions):
                        if 0 <= int(actual) < num_classes and 0 <= int(predicted) < num_classes:
                            confusion_counts[int(actual), int(predicted)] += 1

        # 归一化
        row_sums = confusion_counts.sum(axis=1, keepdims=True)
        confusion_matrix = np.divide(
            confusion_counts,
            row_sums,
            out=np.zeros_like(confusion_counts, dtype=float),
            where=row_sums != 0
        )

        return jsonify({
            'class_names': class_names,
            'confusion_matrix': confusion_matrix.tolist(),
            'confusion_counts': confusion_counts.tolist(),
            'matrix_orientation': 'rows=actual, columns=predicted'
        }), 200

    except Exception as e:
        logger.exception(f"Error generating confusion matrix: {str(e)}")
        return jsonify({'error': str(e)}), 500

@viz_bp.route('/client_distribution', methods=['GET'])
def get_client_distribution():
    """获取客户端数据分布可视化"""
    try:
        if train_routes.fl_system is None or not train_routes.fl_system.clients:
            return jsonify({'error': 'No client data available'}), 400

        # 模拟客户端数据分布
        num_clients = len(train_routes.fl_system.clients)
        class_names = get_dataset_class_names(getattr(train_routes.fl_system, 'dataset_name', 'mnist'))
        num_classes = len(class_names)

        # 生成每个客户端的类别分布
        client_distributions = []
        count_rows = []
        for client in train_routes.fl_system.clients:
            labels = extract_labels(client.train_loader.dataset)
            counts = np.bincount(labels, minlength=num_classes)[:num_classes]
            total = int(counts.sum())
            distribution = (counts / total).tolist() if total else [0] * num_classes
            count_rows.append(counts)
            client_distributions.append({
                'client_id': client.client_id,
                'distribution': distribution,
                'counts': counts.astype(int).tolist(),
                'num_samples': total
            })

        return jsonify({
            'num_clients': num_clients,
            'num_classes': num_classes,
            'class_names': class_names,
            'client_distributions': client_distributions,
            'stats': compute_distribution_stats(np.array(count_rows, dtype=float))
        }), 200

    except Exception as e:
        logger.exception(f"Error getting client distribution: {str(e)}")
        return jsonify({'error': str(e)}), 500

