from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from .logging_config import configure_error_logging

def create_app():
    app = Flask(__name__)
    CORS(app)
    configure_error_logging(app)

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        if isinstance(error, HTTPException):
            if error.code and error.code >= 500:
                app.logger.exception("HTTP exception: %s", error)
            return jsonify({'error': error.description}), error.code
        app.logger.exception("Unhandled backend exception")
        return jsonify({'error': 'Internal server error'}), 500

    # 注册蓝图
    from .routes.main_routes import main_bp
    from .routes.data_routes import data_bp
    from .routes.model_routes import model_bp
    from .routes.train_routes import train_bp
    from .routes.client_routes import client_bp
    from .routes.visualization_routes import viz_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(data_bp, url_prefix='/api/data')
    app.register_blueprint(model_bp, url_prefix='/api/model')
    app.register_blueprint(train_bp, url_prefix='/api/train')
    app.register_blueprint(client_bp, url_prefix='/api/clients')
    app.register_blueprint(viz_bp, url_prefix='/api/viz')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
