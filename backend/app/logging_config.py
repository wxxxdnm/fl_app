import logging
import os
from logging.handlers import RotatingFileHandler


ERROR_LOG_HANDLER_NAME = "fl_app_backend_error_file"
ERROR_LOG_FILENAME = "backend_errors.log"


def get_error_log_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "logs", ERROR_LOG_FILENAME)


def configure_error_logging(app=None):
    log_path = get_error_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root_logger = logging.getLogger()
    abs_log_path = os.path.abspath(log_path)
    for handler in root_logger.handlers:
        if getattr(handler, "name", None) == ERROR_LOG_HANDLER_NAME:
            if app is not None:
                app.config["ERROR_LOG_PATH"] = abs_log_path
            return abs_log_path
        if os.path.abspath(getattr(handler, "baseFilename", "")) == abs_log_path:
            if app is not None:
                app.config["ERROR_LOG_PATH"] = abs_log_path
            return abs_log_path

    handler = RotatingFileHandler(
        abs_log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    handler.set_name(ERROR_LOG_HANDLER_NAME)
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s"
    ))
    root_logger.addHandler(handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.ERROR:
        root_logger.setLevel(logging.ERROR)

    if app is not None:
        app.config["ERROR_LOG_PATH"] = abs_log_path
    return abs_log_path
