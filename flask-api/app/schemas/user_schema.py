from marshmallow import Schema, fields

from marshmallow import Schema, fields, validate, ValidationError

def not_blank(msg=None):
    def validate_not_blank(value):
        if not value or (isinstance(value, str) and not value.strip()):
            raise ValidationError(msg or "Không được trống!")
    return validate_not_blank

class RegisterSchema(Schema):
    username = fields.Str(required=True, validate=not_blank("Tên đăng nhập không được trống!"))
    email = fields.Email(required=True, error_messages={"required": "Email không được trống!", "invalid": "Email không hợp lệ!"})
    name = fields.Str(required=True, validate=not_blank("Tên không được trống!"))
    password = fields.Str(required=True, load_only=True, validate=not_blank("Mật khẩu không được trống!"))

from app.utils.validation import common_rules_validator

EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"

class LoginSchema(Schema):
    email = fields.Str(
        required=False,
        validate=common_rules_validator("Email", {"required": True, "regex": EMAIL_REGEX})
    )
    password = fields.Str(
        required=False,
        load_only=True,
        validate=common_rules_validator("Mật khẩu", {"required": True, "min_length": 6})
    )

class UserSchema(Schema):
    id = fields.Int(dump_only=True)
    username = fields.Str()
    email = fields.Email()
    name = fields.Str()
    created_at = fields.DateTime(dump_only=True)
