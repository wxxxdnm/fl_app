from flask import jsonify, request


def get_json_body():
    """Return a JSON object body or a Flask 400 response tuple."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({'error': 'Request body must be a JSON object'}), 400)
    return data, None
