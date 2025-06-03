from flask import Blueprint, request, jsonify
from app import db
from app.models.user import User
from app.schemas.user_schema import UserSchema, RegisterSchema, LoginSchema
from app.utils.password import hash_password, verify_password
from app.utils.jwt import create_tokens
from app.utils.validation import validate_fields
from flask_jwt_extended import (
    create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
)

user_schema = UserSchema()
auth_bp = Blueprint('auth', __name__)

from app.utils.validation import validate_request
from flask import g

@auth_bp.route('/register', methods=['POST'])
@validate_request(RegisterSchema)
def register():
    data = g.validated_data
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'msg': 'Username already exists'}), 409
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'msg': 'Email already exists'}), 409
    user = User(
        username=data['username'],
        email=data['email'],
        name=data['name'],
        password=hash_password(data['password'])
    )
    db.session.add(user)
    db.session.commit()
    return user_schema.dump(user), 201

@auth_bp.route('/login', methods=['POST'])
@validate_request(LoginSchema)
def login():
    data = g.validated_data
    # Nếu thiếu email hoặc password, trả về lỗi đúng message từ validator (không truy vấn DB)
    missing_fields = {}
    if not data.get('email'):
        missing_fields['email'] = 'Email không được trống!'
    if not data.get('password'):
        missing_fields['password'] = 'Mật khẩu không được trống!'
    if missing_fields:
        return jsonify({'errors': missing_fields, 'message': 'Dữ liệu không hợp lệ'}), 400
    user = User.query.filter_by(email=data['email']).first()
    if not user or not verify_password(data['password'], user.password):
        return jsonify({'msg': 'Invalid credentials'}), 401
    access_token, refresh_token = create_tokens(user.id)
    return jsonify(access_token=access_token, refresh_token=refresh_token), 200

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    identity = get_jwt_identity()
    access_token = create_access_token(identity=identity)
    return jsonify(access_token=access_token), 200
