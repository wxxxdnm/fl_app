from flask import Blueprint, jsonify

from .utils import get_json_body

tuning_bp = Blueprint("tuning", __name__)

tuner = None
applied_best_params = None


def _get_or_create_tuner(config=None, study_name=None):
    global tuner
    from ..services.hyperparameter_tuning import HyperparameterTuner

    if tuner is None or config is not None or study_name is not None:
        tuner = HyperparameterTuner(
            base_config=config or {},
            study_name=study_name or "federated_hyperparameter_tuning",
        )
    return tuner


@tuning_bp.route("/start", methods=["POST"])
def start_tuning():
    try:
        data, error_response = get_json_body()
        if error_response:
            return error_response

        n_trials = int(data.get("n_trials", 20))
        if n_trials < 1:
            return jsonify({"error": "n_trials must be at least 1"}), 400

        config = data.get("config", {})
        if not isinstance(config, dict):
            return jsonify({"error": "config must be an object"}), 400

        study_name = data.get("study_name", "federated_hyperparameter_tuning")
        current_tuner = _get_or_create_tuner(config=config, study_name=study_name)
        return jsonify(current_tuner.start_async(n_trials=n_trials)), 202
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@tuning_bp.route("/status", methods=["GET"])
def tuning_status():
    try:
        current_tuner = _get_or_create_tuner()
        return jsonify(current_tuner.get_status()), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@tuning_bp.route("/history", methods=["GET"])
def tuning_history():
    try:
        current_tuner = _get_or_create_tuner()
        return jsonify({
            "status": current_tuner.get_status(),
            "trials": current_tuner.get_tuning_history(),
        }), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@tuning_bp.route("/best", methods=["GET"])
def tuning_best():
    try:
        current_tuner = _get_or_create_tuner()
        status = current_tuner.get_status()
        return jsonify({
            "best_params": status["best_params"],
            "best_value": status["best_value"],
            "base_config": status["base_config"],
        }), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@tuning_bp.route("/apply-best", methods=["POST"])
def apply_best():
    global applied_best_params
    try:
        current_tuner = _get_or_create_tuner()
        status = current_tuner.get_status()
        if not status["best_params"]:
            return jsonify({"error": "No completed tuning trial is available"}), 400
        applied_best_params = status["best_params"]
        return jsonify({
            "message": "Best hyperparameters are ready to apply",
            "best_params": applied_best_params,
            "best_value": status["best_value"],
        }), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@tuning_bp.route("/suggest", methods=["POST"])
def suggest_hyperparameters():
    try:
        data, error_response = get_json_body()
        if error_response:
            return error_response
        dataset_name = data.get("dataset_name", "mnist")
        current_tuner = _get_or_create_tuner()
        return jsonify(current_tuner.suggest_hyperparameters(dataset_name)), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
