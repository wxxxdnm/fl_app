import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    def __init__(self, features, labels):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        features = features.reshape(features.shape[0], -1)
        labels = np.asarray(labels, dtype=np.int64)
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = labels.tolist()
        self.labels = self.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], torch.tensor(self.targets[index], dtype=torch.long)


class CustomDatasetManager:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        self.upload_dir = os.path.join(data_dir, "uploads")
        self.registry_path = os.path.join(self.upload_dir, "custom_datasets.json")
        os.makedirs(self.upload_dir, exist_ok=True)

    def _read_registry(self) -> Dict:
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_registry(self, registry: Dict):
        with open(self.registry_path, "w", encoding="utf-8") as file:
            json.dump(registry, file, ensure_ascii=False, indent=2)

    def list_datasets(self) -> List[Dict]:
        registry = self._read_registry()
        return [
            item for item in registry.values()
            if item.get("trainable") and os.path.exists(item.get("path", ""))
        ]

    def get_dataset_names(self) -> List[str]:
        return [item["id"] for item in self.list_datasets()]

    def is_custom_dataset(self, dataset_name: str) -> bool:
        return dataset_name in self._read_registry()

    def get_dataset_info(self, dataset_name: str) -> Dict:
        registry = self._read_registry()
        if dataset_name not in registry or not registry[dataset_name].get("trainable"):
            raise ValueError(f"Unsupported custom dataset: {dataset_name}")
        item = registry[dataset_name]
        if not os.path.exists(item.get("path", "")):
            raise ValueError(f"Custom dataset file not found: {dataset_name}")
        return item

    def get_by_filename(self, filename: str) -> Dict:
        for item in self._read_registry().values():
            if item.get("filename") == filename:
                return item
        return None

    def register_file(self, path: str) -> Dict:
        filename = os.path.basename(path)
        record = {
            "filename": filename,
            "path": os.path.abspath(path),
            "trainable": False
        }
        if not self._is_trainable_extension(filename):
            record["parse_error"] = "File type is stored but not trainable"
            return record
        try:
            features, labels, classes = self._load_features_labels(path)
            dataset_id = self._make_dataset_id(filename)
            metadata = {
                "id": dataset_id,
                "name": dataset_id,
                "filename": filename,
                "path": os.path.abspath(path),
                "num_samples": int(len(labels)),
                "num_classes": int(len(classes)),
                "input_shape": [int(np.asarray(features).reshape(len(labels), -1).shape[1])],
                "classes": [str(item) for item in classes],
                "trainable": True,
                "dataset_type": "custom_tabular"
            }
            registry = self._read_registry()
            registry[dataset_id] = metadata
            self._write_registry(registry)
            return metadata
        except Exception as error:
            record["parse_error"] = str(error)
            return record

    def unregister_file(self, filename: str):
        registry = self._read_registry()
        registry = {
            dataset_id: item
            for dataset_id, item in registry.items()
            if item.get("filename") != filename
        }
        self._write_registry(registry)

    def load_dataset(self, dataset_name: str) -> CustomTensorDataset:
        info = self.get_dataset_info(dataset_name)
        features, labels, _ = self._load_features_labels(info["path"])
        return CustomTensorDataset(features, labels)

    def _is_trainable_extension(self, filename: str) -> bool:
        return filename.rsplit(".", 1)[-1].lower() in {"csv", "json", "jsonl", "npy", "npz", "pt", "pth"}

    def _make_dataset_id(self, filename: str) -> str:
        stem = os.path.splitext(filename)[0].lower()
        stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
        return f"custom_{stem}" if not stem.startswith("custom_") else stem

    def _load_features_labels(self, path: str) -> Tuple[np.ndarray, np.ndarray, List]:
        extension = path.rsplit(".", 1)[-1].lower()
        if extension == "csv":
            return self._from_dataframe(pd.read_csv(path))
        if extension == "json":
            return self._from_dataframe(pd.read_json(path))
        if extension == "jsonl":
            return self._from_dataframe(pd.read_json(path, lines=True))
        if extension == "npy":
            return self._from_array(np.load(path, allow_pickle=False))
        if extension == "npz":
            with np.load(path, allow_pickle=False) as data:
                return self._from_npz(data)
        if extension in {"pt", "pth"}:
            return self._from_torch(torch.load(path, map_location="cpu"))
        raise ValueError("Unsupported custom dataset format")

    def _from_dataframe(self, dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List]:
        if dataframe.empty:
            raise ValueError("Dataset file is empty")
        label_column = next((name for name in ("label", "target", "y") if name in dataframe.columns), dataframe.columns[-1])
        labels_raw = dataframe[label_column]
        features = dataframe.drop(columns=[label_column]).apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        return self._finalize(features, labels_raw.to_numpy())

    def _from_npz(self, data) -> Tuple[np.ndarray, np.ndarray, List]:
        keys = set(data.files)
        for feature_key, label_key in (("x", "y"), ("features", "labels"), ("data", "targets")):
            if feature_key in keys and label_key in keys:
                return self._finalize(data[feature_key], data[label_key])
        if len(data.files) >= 2:
            return self._finalize(data[data.files[0]], data[data.files[1]])
        if len(data.files) == 1:
            return self._from_array(data[data.files[0]])
        raise ValueError("NPZ file does not contain arrays")

    def _from_array(self, array) -> Tuple[np.ndarray, np.ndarray, List]:
        array = np.asarray(array)
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("Numpy array must be 2D with features and label in the last column")
        return self._finalize(array[:, :-1], array[:, -1])

    def _from_torch(self, data) -> Tuple[np.ndarray, np.ndarray, List]:
        if isinstance(data, dict):
            keys = set(data.keys())
            for feature_key, label_key in (("x", "y"), ("features", "labels"), ("data", "targets")):
                if feature_key in keys and label_key in keys:
                    return self._finalize(self._to_numpy(data[feature_key]), self._to_numpy(data[label_key]))
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            return self._finalize(self._to_numpy(data[0]), self._to_numpy(data[1]))
        return self._from_array(self._to_numpy(data))

    def _to_numpy(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _finalize(self, features, labels_raw) -> Tuple[np.ndarray, np.ndarray, List]:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        features = features.reshape(features.shape[0], -1)
        labels_raw = np.asarray(labels_raw).reshape(-1)
        if len(features) != len(labels_raw):
            raise ValueError("Features and labels must have the same number of samples")
        if len(labels_raw) < 2:
            raise ValueError("Custom dataset must contain at least 2 samples")
        labels, classes = pd.factorize(labels_raw, sort=False)
        if len(classes) < 2:
            raise ValueError("Custom dataset must contain at least 2 classes")
        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or infinite values")
        return features, labels.astype(np.int64), classes.tolist()


custom_dataset_manager = CustomDatasetManager()
