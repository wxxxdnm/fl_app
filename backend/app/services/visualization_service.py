import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from .data_manager import DataManager
from .federated_learning import calculate_classification_metrics, update_confusion_matrix


def get_dataset_class_names(dataset_name):
    try:
        info = DataManager().get_dataset_info(dataset_name)
        return [str(class_name) for class_name in info['classes']]
    except Exception:
        return [str(i) for i in range(10)]


def extract_labels(dataset):
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
        'client_id': client.client_id,
        'loss': total_loss / max(1, len(client.test_loader)),
        'accuracy': correct / max(1, total),
        'num_samples': total,
        'evaluation_time': evaluation_time,
        'samples_per_second': total / evaluation_time if evaluation_time > 0 else 0.0
    }
    if confusion_matrix is not None:
        metrics.update(calculate_classification_metrics(confusion_matrix))
    return metrics


def get_model_performance_snapshot(fl_system):
    if fl_system is None or not fl_system.clients:
        raise ValueError('No model available')

    client_performance = [
        evaluate_global_model_on_client(fl_system, client)
        for client in fl_system.clients
    ]
    accuracies = [item['accuracy'] for item in client_performance]
    losses = [item['loss'] for item in client_performance]
    precisions = [item.get('precision', 0) for item in client_performance]
    recalls = [item.get('recall', 0) for item in client_performance]
    f1_scores = [item.get('f1_score', 0) for item in client_performance]
    balanced_accuracies = [item.get('balanced_accuracy', 0) for item in client_performance]

    return {
        'client_ids': [item['client_id'] for item in client_performance],
        'accuracies': accuracies,
        'losses': losses,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'balanced_accuracies': balanced_accuracies,
        'samples_per_second': [item.get('samples_per_second', 0) for item in client_performance],
        'sample_sizes': [item['num_samples'] for item in client_performance],
        'stats': {
            'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0,
            'std_accuracy': float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0,
            'mean_loss': float(np.mean(losses)) if losses else 0,
            'std_loss': float(np.std(losses, ddof=1)) if len(losses) > 1 else 0,
            'mean_precision': float(np.mean(precisions)) if precisions else 0,
            'mean_recall': float(np.mean(recalls)) if recalls else 0,
            'mean_f1_score': float(np.mean(f1_scores)) if f1_scores else 0,
            'mean_balanced_accuracy': float(np.mean(balanced_accuracies)) if balanced_accuracies else 0
        }
    }


def get_client_distribution_snapshot(fl_system):
    if fl_system is None or not fl_system.clients:
        raise ValueError('No client data available')

    dataset_name = getattr(fl_system, 'dataset_name', 'mnist')
    class_names = get_dataset_class_names(dataset_name)
    num_classes = len(class_names)
    client_distributions = []
    count_rows = []

    for client in fl_system.clients:
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

    return {
        'num_clients': len(fl_system.clients),
        'num_classes': num_classes,
        'class_names': class_names,
        'client_distributions': client_distributions,
        'stats': compute_distribution_stats(np.array(count_rows, dtype=float))
    }


def get_confusion_matrix_snapshot(fl_system, dataset_name=None):
    if fl_system is None or not fl_system.clients:
        raise ValueError('No model available')

    dataset_name = dataset_name or getattr(fl_system, 'dataset_name', 'mnist')
    class_names = get_dataset_class_names(dataset_name)
    num_classes = len(class_names)
    model = fl_system.global_model
    device = fl_system.device
    model.eval()
    confusion_counts = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for client in fl_system.clients:
            for inputs, targets in client.test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                actuals = targets.cpu().numpy()
                for actual, predicted in zip(actuals, predictions):
                    if 0 <= int(actual) < num_classes and 0 <= int(predicted) < num_classes:
                        confusion_counts[int(actual), int(predicted)] += 1

    row_sums = confusion_counts.sum(axis=1, keepdims=True)
    confusion_matrix = np.divide(
        confusion_counts,
        row_sums,
        out=np.zeros_like(confusion_counts, dtype=float),
        where=row_sums != 0
    )

    return {
        'class_names': class_names,
        'confusion_matrix': confusion_matrix.tolist(),
        'confusion_counts': confusion_counts.tolist(),
        'matrix_orientation': 'rows=actual, columns=predicted'
    }


def get_visualization_snapshot(fl_system, dataset_name=None):
    snapshot = {}
    builders = {
        'model_performance': lambda: get_model_performance_snapshot(fl_system),
        'client_distribution': lambda: get_client_distribution_snapshot(fl_system),
        'confusion_matrix': lambda: get_confusion_matrix_snapshot(fl_system, dataset_name)
    }
    for key, builder in builders.items():
        try:
            snapshot[key] = builder()
        except Exception as error:
            snapshot[f'{key}_error'] = str(error)
    return snapshot
