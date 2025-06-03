from flask import request
from marshmallow import ValidationError
import re

def validate_fields(required_fields, lang='vi'):
    data = request.get_json()
    errors = {}
    messages = {
        'vi': {
            'required': 'Không được trống!',
            'no_data': 'Vui lòng gửi dữ liệu JSON'
        },
        'en': {
            'required': 'This field is required!',
            'no_data': 'Please provide JSON data'
        }
    }
    msg = messages.get(lang, messages['vi'])

    if not data:
        return None, {field: msg['required'] for field in required_fields}, msg['no_data']
    for field in required_fields:
        if field not in data or not data[field]:
            errors[field] = msg['required']
    return data, errors, None

# Common validators for Marshmallow

from functools import wraps
from flask import request, jsonify, g

def validate_request(schema_class):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            json_data = request.get_json() or {}
            schema = schema_class()
            try:
                data = schema.load(json_data)
            except Exception as err:
                errors = err.messages if hasattr(err, 'messages') else str(err)
                return jsonify({'errors': errors, 'message': 'Dữ liệu không hợp lệ'}), 400
            g.validated_data = data
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def not_blank(msg=None):
    def validate_not_blank(value):
        if not value or (isinstance(value, str) and not value.strip()):
            raise ValidationError(msg or "Không được trống!")
    return validate_not_blank

def min_length(min_len, msg=None):
    def validate_min_length(value):
        if value is None or len(value) < min_len:
            raise ValidationError(msg or f"Phải có ít nhất {min_len} ký tự!")
    return validate_min_length

def max_length(max_len, msg=None):
    def validate_max_length(value):
        if value is not None and len(value) > max_len:
            raise ValidationError(msg or f"Không quá {max_len} ký tự!")
    return validate_max_length

def regex_match(pattern, msg=None):
    def validate_regex(value):
        if not re.match(pattern, value or ""):
            raise ValidationError(msg or "Không đúng định dạng!")
    return validate_regex

# --- Common rules validator (configurable, message tự động) ---
def common_rules_validator(field_title, rules):
    """
    rules: dict, ví dụ:
    {
        "required": True,
        "min_length": 6,
        "max_length": 30,
        "regex": r"^[a-zA-Z0-9_]+$"
    }
    """
    def validator(value):
        # required
        if rules.get("required", False):
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValidationError(f"{field_title} không được trống!")
        # min_length
        min_len = rules.get("min_length")
        if min_len is not None and value is not None and len(value) < min_len:
            raise ValidationError(f"{field_title} tối thiểu {min_len} ký tự!")
        # max_length
        max_len = rules.get("max_length")
        if max_len is not None and value is not None and len(value) > max_len:
            raise ValidationError(f"{field_title} tối đa {max_len} ký tự!")
        # regex
        pattern = rules.get("regex")
        if pattern and value is not None and not re.match(pattern, value):
            raise ValidationError(f"{field_title} không đúng định dạng!")
    return validator
