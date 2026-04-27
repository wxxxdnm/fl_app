import copy
import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

import optuna
import torch

from ..services.data_manager import DataManager
from ..services.federated_learning import FLClient, FederatedLearning
from ..services.model_manager import ModelManager

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    def __init__(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        storage_url: Optional[str] = None,
        study_name: str = "federated_hyperparameter_tuning",
    ):
        self.base_config = self._normalize_config(base_config or {})
        self.model_manager = ModelManager()
        self.study_name = study_name
        self.storage_url = storage_url or self._default_storage_url()
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.status = "Not started"
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @staticmethod
    def _default_storage_url() -> str:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return f"sqlite:///{os.path.join(root, 'tuning_studies.db')}"

    @staticmethod
    def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "dataset_name": config.get("dataset_name", "mnist"),
            "num_clients": int(config.get("num_clients", 10)),
            "trial_rounds": int(config.get("trial_rounds", 3)),
            "iid": bool(config.get("iid", True)),
            "device": config.get("device", "cuda"),
            "aggregation_algorithm": str(config.get("aggregation_algorithm", "fedavg")).lower(),
            "data_dir": config.get("data_dir"),
            "seed": int(config.get("seed", 42)),
            "tune_num_clients": bool(config.get("tune_num_clients", False)),
        }
        if normalized["num_clients"] < 1:
            raise ValueError("num_clients must be at least 1")
        if normalized["trial_rounds"] < 1:
            raise ValueError("trial_rounds must be at least 1")
        if normalized["device"] == "cuda" and not torch.cuda.is_available():
            normalized["device"] = "cpu"
        if normalized["aggregation_algorithm"] not in FederatedLearning.SUPPORTED_ALGORITHMS:
            supported = ", ".join(sorted(FederatedLearning.SUPPORTED_ALGORITHMS))
            raise ValueError(f"Unsupported aggregation_algorithm. Supported: {supported}")
        return normalized

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        algorithm = self.base_config["aggregation_algorithm"]
        params = {
            "client_lr": trial.suggest_float("client_lr", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "client_fraction": trial.suggest_float("client_fraction", 0.3, 1.0),
            "local_epochs": trial.suggest_int("local_epochs", 1, 5),
            "optimizer": trial.suggest_categorical("optimizer", ["SGD", "Adam"]),
        }

        if self.base_config["tune_num_clients"]:
            params["num_clients"] = trial.suggest_int("num_clients", 5, 20)
        else:
            params["num_clients"] = self.base_config["num_clients"]

        if algorithm in ("fedavgm", "fedadam", "fedyogi", "fedadagrad"):
            params["server_lr"] = trial.suggest_float("server_lr", 0.05, 2.0, log=True)
        else:
            params["server_lr"] = 1.0

        if algorithm == "fedavgm":
            params["server_momentum"] = trial.suggest_float("server_momentum", 0.0, 0.99)
        else:
            params["server_momentum"] = 0.9

        if algorithm == "fedprox":
            params["proximal_mu"] = trial.suggest_float("proximal_mu", 1e-4, 1.0, log=True)
        else:
            params["proximal_mu"] = 0.0

        if algorithm in ("fedadam", "fedyogi", "fedadagrad"):
            params["adaptive_beta1"] = trial.suggest_float("adaptive_beta1", 0.0, 0.99)
            params["adaptive_beta2"] = trial.suggest_float("adaptive_beta2", 0.0, 0.999)
            params["adaptive_tau"] = trial.suggest_float("adaptive_tau", 1e-6, 1e-2, log=True)
        else:
            params["adaptive_beta1"] = 0.9
            params["adaptive_beta2"] = 0.99
            params["adaptive_tau"] = 1e-3

        return params

    def _create_client_optimizer(self, client: FLClient, params: Dict[str, Any]):
        if params["optimizer"] == "Adam":
            return torch.optim.Adam(client.model.parameters(), lr=params["client_lr"])
        return torch.optim.SGD(client.model.parameters(), lr=params["client_lr"], momentum=0.9)

    def _build_trial_system(self, params: Dict[str, Any], trial_number: int) -> FederatedLearning:
        seed = self.base_config["seed"] + trial_number
        random.seed(seed)
        torch.manual_seed(seed)

        dataset_name = self.base_config["dataset_name"]
        global_model = self.model_manager.create_model(dataset_name)
        fl_system = FederatedLearning(
            global_model,
            self.base_config["device"],
            aggregation_algorithm=self.base_config["aggregation_algorithm"],
            server_lr=params["server_lr"],
            server_momentum=params["server_momentum"],
            proximal_mu=params["proximal_mu"],
            adaptive_beta1=params["adaptive_beta1"],
            adaptive_beta2=params["adaptive_beta2"],
            adaptive_tau=params["adaptive_tau"],
        )
        fl_system.dataset_name = dataset_name

        data_manager = DataManager(data_dir=self.base_config["data_dir"])
        client_dataloaders = data_manager.create_federated_datasets(
            dataset_name=dataset_name,
            num_clients=params["num_clients"],
            batch_size=params["batch_size"],
            iid=self.base_config["iid"],
        )

        for client_id in range(params["num_clients"]):
            client = FLClient(
                client_id,
                global_model,
                client_dataloaders[f"client_{client_id}_train"],
                client_dataloaders[f"client_{client_id}_test"],
                self.base_config["device"],
            )
            client.local_epochs = params["local_epochs"]
            client.optimizer = self._create_client_optimizer(client, params)
            fl_system.add_client(client)

        return fl_system

    def objective(self, trial: optuna.Trial) -> float:
        params = self._sample_params(trial)
        fl_system = self._build_trial_system(params, trial.number)
        num_clients = len(fl_system.clients)
        num_selected = max(1, int(params["client_fraction"] * num_clients))
        generator = torch.Generator().manual_seed(self.base_config["seed"] + trial.number)
        recent_scores: List[float] = []

        for round_idx in range(self.base_config["trial_rounds"]):
            indices = torch.randperm(num_clients, generator=generator)[:num_selected].tolist()
            selected_clients = [fl_system.clients[i] for i in indices]
            fl_system.train_round(selected_clients)
            metrics = fl_system.evaluate_global_model(fl_system.clients)
            score = float(metrics["accuracy"])
            recent_scores.append(score)
            trial.report(score, round_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return sum(recent_scores[-3:]) / min(3, len(recent_scores))

    def tune(self, n_trials: int = 20) -> Dict[str, Any]:
        logger.info("Starting hyperparameter tuning with %s trials", n_trials)
        self.status = "Running"
        self.error = None
        self.started_at = time.time()
        self.finished_at = None

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.base_config["seed"]),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=True,
        )
        self.study.set_user_attr("base_config", copy.deepcopy(self.base_config))
        self.study.optimize(self.objective, n_trials=n_trials)

        try:
            self.best_params = self.study.best_params
        except ValueError:
            self.best_params = None
        self.status = "Completed"
        self.finished_at = time.time()
        return self.get_status()

    def start_async(self, n_trials: int = 20) -> Dict[str, Any]:
        with self._lock:
            if self.status == "Running":
                raise RuntimeError("Hyperparameter tuning is already running")
            self._thread = threading.Thread(target=self._run_async, args=(n_trials,), daemon=True)
            self._thread.start()
        return self.get_status()

    def _run_async(self, n_trials: int):
        try:
            self.tune(n_trials=n_trials)
        except Exception as exc:
            logger.exception("Hyperparameter tuning failed")
            self.status = "Error"
            self.error = str(exc)
            self.finished_at = time.time()

    def get_status(self) -> Dict[str, Any]:
        study = self.study or self._load_study_if_available()
        best_value = None
        best_params = self.best_params
        completed_trials = 0
        total_trials = 0
        if study is not None:
            total_trials = len(study.trials)
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            if completed_trials:
                best_value = float(study.best_value)
                best_params = study.best_params

        return {
            "status": self.status,
            "error": self.error,
            "study_name": self.study_name,
            "base_config": self.base_config,
            "total_trials": total_trials,
            "completed_trials": completed_trials,
            "best_value": best_value,
            "best_params": best_params,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def _load_study_if_available(self) -> Optional[optuna.Study]:
        try:
            self.study = optuna.load_study(study_name=self.study_name, storage=self.storage_url)
            return self.study
        except KeyError:
            return None

    def get_tuning_history(self) -> List[Dict[str, Any]]:
        study = self.study or self._load_study_if_available()
        if study is None:
            return []

        history = []
        for trial in study.trials:
            history.append({
                "trial": trial.number,
                "params": trial.params,
                "value": None if trial.value is None else float(trial.value),
                "state": trial.state.name,
                "intermediate_values": {
                    str(step): float(value)
                    for step, value in trial.intermediate_values.items()
                },
            })
        return history

    def suggest_hyperparameters(self, dataset_name: str) -> Dict[str, Any]:
        recommendations = {
            "mnist": {
                "client_lr": 0.01,
                "batch_size": 64,
                "num_clients": 10,
                "client_fraction": 0.5,
                "local_epochs": 3,
                "optimizer": "SGD",
            },
            "cifar10": {
                "client_lr": 0.001,
                "batch_size": 128,
                "num_clients": 10,
                "client_fraction": 0.7,
                "local_epochs": 3,
                "optimizer": "Adam",
            },
        }
        return recommendations.get(dataset_name, recommendations["mnist"])
