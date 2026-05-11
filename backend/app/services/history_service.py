import datetime
import json
import os
from typing import Dict, List

import torch


class HistoryService:
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "history"))
        self.storage_dir = storage_dir
        self.training_history_path = os.path.join(storage_dir, "training_runs.json")
        self.model_history_path = os.path.join(storage_dir, "model_records.json")
        self.checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints"))
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _read_json(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_json(self, path: str, records: List[Dict]):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)

    def _to_json_safe(self, value):
        if isinstance(value, dict):
            return {str(key): self._to_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_json_safe(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value

    def add_training_run(
        self,
        config: Dict,
        history: List[Dict],
        status: str = "Completed",
        error: str = None,
        visualization: Dict = None
    ) -> Dict:
        records = self._read_json(self.training_history_path)
        safe_history = self._to_json_safe(history)
        safe_visualization = self._to_json_safe(visualization or {})
        final_metrics = safe_history[-1].get("global_metrics", {}) if safe_history else {}
        timestamp = datetime.datetime.now().isoformat()
        record = {
            "id": timestamp.replace(":", "-").replace(".", "-"),
            "timestamp": timestamp,
            "status": status,
            "dataset_name": config.get("dataset_name"),
            "model_name": config.get("model_name"),
            "num_clients": config.get("num_clients"),
            "num_rounds": config.get("num_rounds"),
            "client_fraction": config.get("client_fraction"),
            "aggregation_algorithm": config.get("aggregation_algorithm"),
            "iid": config.get("iid"),
            "non_iid_alpha": config.get("non_iid_alpha"),
            "non_iid_seed": config.get("non_iid_seed"),
            "device": config.get("device"),
            "batch_size": config.get("batch_size"),
            "server_lr": config.get("server_lr"),
            "server_momentum": config.get("server_momentum"),
            "proximal_mu": config.get("proximal_mu"),
            "adaptive_beta1": config.get("adaptive_beta1"),
            "adaptive_beta2": config.get("adaptive_beta2"),
            "adaptive_tau": config.get("adaptive_tau"),
            "rounds": len(safe_history),
            "final_accuracy": final_metrics.get("accuracy", 0),
            "final_loss": final_metrics.get("loss", 0),
            "final_f1_score": final_metrics.get("f1_score", 0),
            "history": safe_history,
            "visualization": safe_visualization,
            "error": error
        }
        records.insert(0, record)
        self._write_json(self.training_history_path, records[:50])
        return record

    def get_training_runs(self, limit: int = 20) -> List[Dict]:
        return self._read_json(self.training_history_path)[:limit]

    def get_latest_training_run(self) -> Dict:
        records = self.get_training_runs(1)
        return records[0] if records else None

    def delete_training_run(self, run_id: str) -> bool:
        records = self._read_json(self.training_history_path)
        remaining_records = [record for record in records if record.get("id") != run_id]
        if len(remaining_records) == len(records):
            return False
        self._write_json(self.training_history_path, remaining_records)
        return True

    def add_model_record(self, path: str, metadata: Dict = None) -> Dict:
        records = self._read_json(self.model_history_path)
        abs_path = os.path.abspath(path)
        stat = os.stat(abs_path) if os.path.exists(abs_path) else None
        timestamp = datetime.datetime.now().isoformat()
        record = {
            "id": timestamp.replace(":", "-").replace(".", "-"),
            "timestamp": timestamp,
            "path": abs_path,
            "filename": os.path.basename(abs_path),
            "size_bytes": stat.st_size if stat else 0,
            **(metadata or {})
        }
        records = [item for item in records if os.path.abspath(item.get("path", "")) != abs_path]
        records.insert(0, self._to_json_safe(record))
        self._write_json(self.model_history_path, records[:100])
        return record

    def get_model_records(self, limit: int = 50) -> List[Dict]:
        records = self._read_json(self.model_history_path)
        by_path = {os.path.abspath(item.get("path", "")): item for item in records if item.get("path")}
        for root, _, files in os.walk(self.checkpoint_dir):
            for filename in files:
                if not filename.endswith((".pth", ".pt")):
                    continue
                path = os.path.abspath(os.path.join(root, filename))
                if path not in by_path:
                    by_path[path] = self._checkpoint_record(path)
        by_path = {
            path: self._enrich_model_record(record, path)
            for path, record in by_path.items()
        }
        sorted_records = sorted(
            by_path.values(),
            key=lambda item: item.get("timestamp") or item.get("modified_time") or "",
            reverse=True
        )
        return sorted_records[:limit]

    def _enrich_model_record(self, record: Dict, path: str) -> Dict:
        if record.get("dataset_name") and record.get("model_name"):
            return record
        if not os.path.exists(path):
            return record
        checkpoint_record = self._checkpoint_record(path)
        enriched = {**checkpoint_record, **record}
        for key in ("dataset_name", "model_name", "model_class", "num_classes", "rounds", "aggregation_algorithm", "num_clients"):
            if not enriched.get(key) and checkpoint_record.get(key):
                enriched[key] = checkpoint_record[key]
        return enriched

    def _checkpoint_record(self, path: str) -> Dict:
        stat = os.stat(path)
        record = {
            "id": path,
            "path": path,
            "filename": os.path.basename(path),
            "size_bytes": stat.st_size,
            "modified_time": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
        try:
            checkpoint = torch.load(path, map_location="cpu")
            if isinstance(checkpoint, dict):
                metadata = checkpoint.get("metadata", {}) or {}
                inferred = self._infer_checkpoint_info(path, checkpoint)
                record.update({
                    "model_class": checkpoint.get("model_class") or inferred.get("model_class"),
                    "dataset_name": checkpoint.get("dataset_name") or inferred.get("dataset_name"),
                    "model_name": checkpoint.get("model_name") or inferred.get("model_name"),
                    "num_classes": checkpoint.get("num_classes") or inferred.get("num_classes"),
                    "rounds": metadata.get("rounds"),
                    "aggregation_algorithm": metadata.get("aggregation_algorithm"),
                    "num_clients": metadata.get("num_clients")
                })
        except Exception:
            record["load_error"] = True
        return self._to_json_safe(record)

    def _infer_checkpoint_info(self, path: str, checkpoint: Dict) -> Dict:
        filename = os.path.basename(path).lower()
        dataset_name = None
        for candidate in ("cifar100", "cifar10", "mnist"):
            if candidate in filename:
                dataset_name = candidate
                break

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()
        output_dim = None
        for key in ("fc2.weight", "classifier.4.weight", "classifier.3.weight", "layers.6.weight"):
            weight = state_dict.get(key) if isinstance(state_dict, dict) else None
            if hasattr(weight, "shape") and len(weight.shape) > 0:
                output_dim = int(weight.shape[0])
                break

        if output_dim == 100:
            dataset_name = dataset_name or "cifar100"
        elif output_dim == 10 and dataset_name is None and ("conv3.weight" in keys or "bn3.weight" in keys):
            dataset_name = "cifar10"
        elif output_dim == 10 and dataset_name is None:
            dataset_name = "mnist"

        model_name = None
        model_class = None
        if {"conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight"}.issubset(keys):
            model_name = "cnn"
            if dataset_name == "cifar100":
                model_class = "CIFAR100Net"
            elif dataset_name == "cifar10" or "conv3.weight" in keys:
                model_class = "CIFAR10Net"
            else:
                model_class = "MNISTNet"
        elif any(key.startswith("layers.") for key in keys):
            model_name = "mlp"
            model_class = "MLPNet"
        elif any(key.startswith("layer") for key in keys):
            model_name = "resnet"
            model_class = "CIFARResNet"
        elif any(key.startswith("features.") for key in keys):
            model_name = "deep_cnn" if dataset_name in ("cifar10", "cifar100") else "lenet"
            model_class = "CIFARDeepCNN" if model_name == "deep_cnn" else "LeNetNet"

        return {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "model_class": model_class,
            "num_classes": output_dim
        }


history_service = HistoryService()
